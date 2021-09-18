import os 

import numpy as np 
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
# import pandas as pd
import copy


def get_dirty_ids(sensor_data):
    raw_data = copy.deepcopy(sensor_data)

    # Remove nans
    raw_data = np.where(np.isnan(raw_data), 1e10, raw_data)
    # Remove rows with data > 1e4

    filter_ids = np.invert(np.bitwise_or.reduce(np.abs(raw_data)>1e6,axis=1))
    return filter_ids

def remove_temp(sensor_data,first_temp_id=0):
    temp_mask = np.ones((sensor_data.shape[1],), dtype=bool)
    temp_mask[first_temp_id:first_temp_id+20:4] = False
    temp_removed = sensor_data[...,temp_mask]
    return temp_removed

def interp_robot_data(robot_data, sensor_times):
    interp_thres = 3.
    interp_data = []
    # For interpolation, we will use robot_data for endpoints and interpolate robot
    # data for points in between
    curr_sr = 0
    chunkr_st = 0
    chunkr_e = 0

    chunks_st = 0
    chunks_e = 0
    sensor_mask = np.zeros_like(sensor_times,dtype=bool)

    for j in range(robot_data.shape[0]):
        if j == 0:
            continue
        # First detect chunks
        # Check if x coordinate is within threshold
        if j < robot_data.shape[0]-1:
            if abs(robot_data[j,1] - robot_data[j-1,1]) < interp_thres:
                continue

        chunkr_e = j
        time_offset = robot_data[chunkr_st,0]

        # Identify corresponding chunks in the sensor data
        while sensor_times[curr_sr] <= robot_data[chunkr_st,0]:
            curr_sr+=1
        chunks_st = curr_sr
        
        while sensor_times[curr_sr] <= robot_data[chunkr_e-1,0]:
            curr_sr+=1
        chunks_e = curr_sr

        # Interpolate between chunks for sensor data
        interp_f = interp1d(
            robot_data[chunkr_st:chunkr_e,0] - time_offset, 
            robot_data[chunkr_st:chunkr_e,1:],axis=0)
        
        int_robot_data = interp_f(sensor_times[chunks_st:chunks_e] - time_offset)
        sensor_mask[chunks_st:chunks_e] = True
        # print(chunkr_e - chunkr_st, chunks_e - chunks_st, sensor_times[chunks_e] - time_offset, robot_data[chunkr_e,0] - time_offset)
        chunkr_st = chunkr_e
        
        int_robot_data = np.around(int_robot_data,decimals=1)
        # Add data to interp_data
        interp_data += [int_robot_data]

    return np.concatenate(interp_data, axis=0), sensor_mask

class MagneticSensorData(Dataset):
    def __init__(self, data_dirs, expt_id, scale_std=1, 
        sensor_std = None, force_std=None, zero_comp='none',
        force_lims=None, append_time=False, skip_cent_mag=False,
        cent_mag_only=False, return_nbd=False, append_zeros=False,
        data_limit=98375):
        '''

        '''
        super(Dataset, self).__init__()
        self.data_dirs = data_dirs
        self.expt_id = expt_id
        self.scale_std = scale_std
        self.sensor_std = sensor_std
        self.force_std = force_std
        self.zero_comp = zero_comp
        self.force_lims = force_lims
        self.append_time = append_time
        self.skip_cent_mag = skip_cent_mag
        self.cent_mag_only = cent_mag_only

        assert (cent_mag_only and skip_cent_mag) != 1

        self.robot_data = []
        self.sensor_data = []
        self.force_data = []
        self.zero_means = []
        self.ds_lengths = []

        self.dps_per_pt = 5
        self.grid_size = 65
        self.num_depths = 6

        self.grid_pts = self.dps_per_pt * self.grid_size
        self.gridset_size = self.grid_pts * self.num_depths

        self.return_nbd = return_nbd
        self.append_zeros = append_zeros

        for d in data_dirs:
            if expt_id == 2:
                read_data = np.loadtxt(os.path.join(d,'aggr_data.txt'))
                if data_limit is not None:
                    data = read_data[:98375]
                else:
                    data = read_data
                raw_sensor_data = data[...,3:23]
                filter_ids = get_dirty_ids(raw_sensor_data)
                
                robot_data = data[filter_ids,:3]
                sensor_data = data[filter_ids,3:23]
                if data.shape[-1] == 26:
                    force_data = data[filter_ids,-3:]
                else:
                    print('Warning: Force data missing')
                    force_data = np.zeros((robot_data.shape[0],3))
            elif expt_id == 3:
                raw_robot_data = np.loadtxt(os.path.join(d,'robot_data.txt'))
                raw_sensor_data = np.loadtxt(os.path.join(d,'sensor_data.txt'))

                # First column of raw sensor data containes timestamps
                filter_ids = get_dirty_ids(raw_sensor_data[...,1:])
                
                # Filter out times, sensor data and force data, if available
                filt_times = raw_sensor_data[filter_ids,0]
                sensor_data = raw_sensor_data[filter_ids,1:21]
                if raw_sensor_data.shape[-1] == 24:
                    force_data = raw_sensor_data[filter_ids,-3:]
                else:
                    print('Warning: Force data missing')
                    force_data = np.zeros((sensor_data.shape[0],3))
                # Use timestamps from sensor data to linearly interpolate robot 
                # kinematic data
                robot_data, sensor_mask = interp_robot_data(raw_robot_data, filt_times)
                sensor_data = sensor_data[sensor_mask]
                force_data = force_data[sensor_mask]
            
            # Find zero_mean for that sensor and subtract from sensor_data
            zero_ids = self.filt_pts([0.,0., None], robot_data)
            ind_zero_ids = self.filt_pts([None, None, 10.0], robot_data)
            nonzero_ids = np.invert(np.bitwise_or(zero_ids, ind_zero_ids))

            if force_lims is not None:
                force_lim_ids = np.bitwise_and(
                    -force_data[:,-1] >= force_lims[0], 
                    -force_data[:,-1] <= force_lims[1])
            else:
                force_lim_ids = np.ones((force_data.shape[0],), 
                    dtype=bool)

            zero_mean = np.mean(sensor_data[zero_ids], axis = 0, keepdims=True)
            zero_mean_f = np.mean(force_data[zero_ids], axis = 0, keepdims=True)

            raw_sensor_data = sensor_data.copy()

            if zero_comp == 'agg':
                sensor_data = sensor_data - zero_mean
            
            if zero_comp == 'perdepth10':
                num_dps = 5
                ids_list = np.nonzero(zero_ids)[0][::num_dps*10]
                for i, ind in enumerate(ids_list):
                    # if i == ids_list.shape[0] - 1
                    # print(i,ind)
                    curr_mean = np.mean(sensor_data[ind:ind+num_dps],axis=0,keepdims=True)
                    if i < ids_list.shape[0] - 1:
                        sensor_data[ind:ids_list[i+1]] = sensor_data[ind:ids_list[i+1]] - curr_mean
                    else:
                        sensor_data[ind:-1] = sensor_data[ind:-1] - curr_mean

            if zero_comp == 'imm':
                sensor_data = sensor_data - np.roll(sensor_data,5,axis=0)
            

            if zero_comp == 'init':
                sensor_data = sensor_data - np.mean(sensor_data[zero_ids][:5],axis=0,keepdims=True)
            
            self.zero_means += [remove_temp(zero_mean)]

            # if sensor_std is None:
            #     self.sensor_std = np.std(self.sensor_data, axis=0, keepdims = True)
            # self.sensor_data = self.sensor_data / (self.sensor_std * scale_std)
            # Only keep interaction data, and delete zero data
            nonzero_sensor_data = remove_temp(sensor_data[nonzero_ids])
            if append_time:
                times = np.linspace(0,1,len(nonzero_sensor_data)//5)
                times = np.repeat(times, 5)
                nonzero_sensor_data = np.concatenate(
                    (nonzero_sensor_data, times[:,None]), axis=-1)
            filtered_sensor_data = nonzero_sensor_data[force_lim_ids[nonzero_ids]] 
            if append_zeros:
                zero_data = remove_temp(raw_sensor_data[np.roll(nonzero_ids, -5)])
                filtered_zero_data = zero_data[force_lim_ids[nonzero_ids]]
                filtered_sensor_data = np.concatenate((filtered_sensor_data, filtered_zero_data),axis=-1)


            self.robot_data += [robot_data[nonzero_ids*force_lim_ids]]
            self.sensor_data += [filtered_sensor_data]
            self.force_data += [force_data[nonzero_ids*force_lim_ids]]
            self.ds_lengths += [self.robot_data[-1].shape[0]]
        print(self.ds_lengths)
        self.robot_data = np.concatenate(self.robot_data)
        self.robot_data[:,-1] += 2.1
        self.sensor_data = np.concatenate(self.sensor_data)
        self.force_data = np.concatenate(self.force_data)

        if sensor_std is None:
            self.sensor_std = np.std(self.sensor_data, axis=0, keepdims = True)
            # self.sensor_std = 5*np.std(self.sensor_data, axis=0, keepdims=True)
        self.sensor_data = self.sensor_data / (self.sensor_std * scale_std)

        if skip_cent_mag:
            self.sensor_data = self.sensor_data[...,3:]
        if cent_mag_only:
            self.sensor_data = self.sensor_data[...,:3]
            
        if force_std is None:
            self.force_std = np.std(self.force_data, axis=0, keepdims = True)
        # self.force_data = self.force_data / (self.force_std * scale_std)
        print(np.max(self.force_data, axis=0))
        print('Force std:', self.force_std)
        print('Sensor max:', np.amax(self.sensor_data, axis=0))
        print('Sensor std:', self.sensor_std)

    def __len__(self):
        return self.sensor_data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {
            'robot':self.robot_data[idx], 
            'sensor':self.sensor_data[idx],
            'force':self.force_data[idx],}
            # 'sensor_nbd': self.sensor_data[idx::325],
            # 'force_nbd':self.force_data[idx::325]}
        if self.return_nbd:
            assert self.force_lims is None
            # This corresponds to which part of the 
            set_id = idx % (self.gridset_size)
            grid_id = set_id // (self.grid_pts)
            # close_offset = 1 if (grid_id <self.num_depths-1) else -1
            # dist_offset = 2 if (grid_id <self.num_depths-2) else -2
            # sample['sensor_close'] = self.sensor_data[idx + close_offset *self.grid_pts]
            # sample['sensor_distant'] = self.sensor_data[idx + dist_offset *self.grid_pts]

            # This is just for degubbing
            sample['robot_nbd'] = self.robot_data[idx - set_id:idx - set_id + self.gridset_size:self.grid_pts]
            sample['sensor_nbd'] = self.sensor_data[idx - set_id:idx - set_id + self.gridset_size:self.grid_pts]
            sample['force_nbd'] = self.force_data[idx - set_id:idx - set_id + self.gridset_size:self.grid_pts]
            sample['grid_id'] = grid_id
        return sample
    
    def filt_pts(self, pt, robot_data=None):
        if torch.is_tensor(pt):
            pt = pt.numpy()
        if isinstance(pt, list):
            pt = np.array(pt)
        
        if robot_data is None:
            robot_data = self.robot_data

        filt_mask = [c is None for c in pt]
        rec_pt = np.array([p if p is not None else 0 for p in pt])
        reqd_ids = np.bitwise_and.reduce(
            np.bitwise_or(
                np.isclose(robot_data, rec_pt[None,:]), np.asarray(filt_mask)[None,:]), axis=1
        )
        
        # reqd_ids = np.bitwise_and.reduce(
        #             np.isclose(robot_data[...,:pt.shape[-1]], pt[None,:]), axis=1
        #         )
        return reqd_ids

if __name__ == '__main__':
    expt_id = 3
    expt_dict = {
        2: 'const_depth_snake',
        3:'shear_lines'
    }
    data_root = os.path.join('./expts',expt_dict[expt_id])
    
    data_dirs = ['2021-02-17_23-29-26']
    sensor_ids = [1,2,3,4]
    dd_list = []
    for d in data_dirs:
        for sid in sensor_ids:
            curr_path = os.path.join(data_root,d,'sensor_'+str(sid))
            if os.path.exists(curr_path):
                dd_list.append(curr_path)            
    # data = MagneticSensorData(dd_list, expt_id)
    data = MagneticSensorData_Unsup(dd_list[:2], dd_list[2:], expt_id)
    print(data.robot_data.shape, data.sensor_data.shape, data.force_data.shape, data.unlabeled_masks.shape)
    print(data.zero_means)