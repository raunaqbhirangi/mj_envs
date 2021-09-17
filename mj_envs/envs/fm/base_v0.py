from time import process_time
import gym
import numpy as np
from mj_envs.envs import env_base
# from mj_envs.utils.xml_utils import reassign_parent
import os
import collections
import torch
import yaml

from .reskin_files.train import ReSkinSim
import ipdb

class DManusBase(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'mag'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": -1.0,
        "bonus": 4.0,
        "penalty": -50,
    }

    def __init__(self, 
        model_path, 
        config_path, 
        obs_keys=DEFAULT_OBS_KEYS,
        **kwargs):

        curr_dir = os.path.dirname(os.path.abspath(__file__))

        # self.target_pose = target_pose
        # self.use_mags = use_mags
        

        env_base.MujocoEnv.__init__(self,
                                curr_dir+model_path)
        self.obs_keys = obs_keys
        
        # Find number of magnetometers
        num_sites = len(self.sim.data.site_xpos)
        site_names = list(map(self.sim.model.site_id2name, [i for i in range(num_sites)]))
        self.mag_mask = [True if 'mag' in site_names[i] else False for i in range(num_sites)]
        self.mag_names = [name for name in site_names if 'mag' in name]
        
        self.site_colors = self.sim.model.site_rgba.copy()
        self.mag_model = ReSkinSim()
        self.mag_model.load_state_dict(
            torch.load(curr_dir+'/reskin_files/reskin_sim_weights'))
        with open(curr_dir + '/reskin_files/model_config.yaml','r') as f:
            self.mag_model_config = yaml.safe_load(f)
        
        self._setup(obs_keys=self.obs_keys,
            weighted_reward_keys = self.DEFAULT_RWD_KEYS_AND_WEIGHTS,
            reward_mode = "dense",             
            frame_skip = 40,
            normalize_act = True,
            # obs_range = (-10, 10),
            # seed = None,
            act_mode = "pos",
            is_hardware = False,
            config_path = config_path,
            # rwd_viz = False,
            robot_name = 'dmanus')
    
    def _setup(self,**kwargs):
        env_base.MujocoEnv._setup(self, **kwargs)

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
        obs_dict['pose_err'] = obs_dict['qp'] - self.target_pose
        obs_dict['mag'] = self.get_mag_obs(sim)

        return obs_dict
    
    def get_mag_obs(self, sim):
        '''
        Use simulation data to identify contacts. Use contact location and force 
        to simulate magnetic field data from ReSkin. 

        Parameters
        ----------
        sim: MjSim object
        '''

        num_mags = len(self.mag_names)
        mag_obs = np.zeros((num_mags, 3))

        # Create a copy of original colors
        colors = self.site_colors.copy()

        # Detection threshold distance for magnetometer
        # (16 mm Manhattan distance)
        mag_thres = 0.035

        # Check for contacts
        for con_id in range(sim.data.ncon):
            c = sim.data.contact[con_id]

            # Get ids for the colliding bodies
            body1_id = sim.model.geom_bodyid[c.geom1]
            body2_id = sim.model.geom_bodyid[c.geom2]
            
            body2_xmat = sim.data.body_xmat[body2_id].reshape((3,3))
            # For now just assumes that the second geom is on the hand.
            # TODO: Ask vikash how to check if it belongs to dmanus.
            # ipdb.set_trace()

            # Filter sites with body id matching collision body and make 
            # sure they're magnetometers
            sites_mask = (sim.model.site_bodyid == body2_id) * self.mag_mask
            
            # Find site closest to contact point, if there are any
            # sites on the body
            if np.sum(sites_mask) != 0:
                sites_r = c.pos - sim.data.site_xpos[sites_mask]
                sites_r_local = sites_r @ body2_xmat

                sites_dist = np.linalg.norm(sites_r_local[...,::2], ord=2, axis=-1)
                # sites_mandist = np.linalg.norm(sites_r, ord=1, axis=-1)
                
                # v0.1: Just light up the closest magnetometer  
                closest_site = np.argmin(sites_dist)
                closest_site_id = np.arange(len(sites_mask))[sites_mask][closest_site]
            
                # # Change color to red
                # colors[closest_site_id] = [1.,0.,0.,1.]

                # v1.0: Predict magnetic field for all magnetometers within
                # range
                mags_in_range = sites_dist < mag_thres
                mags_in_range_ids = np.arange(len(sites_mask))[sites_mask][mags_in_range]
                # Need to project sites_r to mag coordinate frame
                # ipdb.set_trace()

                contact_location_xy = sites_r_local[...,::2]
                mag_output = self.get_mag_output(contact_location_xy, mags_in_range, mag_thres)

                mag_obs[sites_mask[self.mag_mask]] = mag_output

                # Color magnetometers based on outputs
                mags_in_range_ids = np.arange(len(sites_mask))[sites_mask][mags_in_range]
                colors[mags_in_range_ids,:3] = np.clip(0.5 + mag_output[mags_in_range]/2,-1.,1.)

        self.sim.model.site_rgba[:,:] = colors

        return mag_obs.reshape((-1,))

    def get_mag_output(self, contact_location_xy, mags_in_range, mag_thres):
        # Default scaling for input distances is 8mm. Magnetometers here are much further apart.
        # So we use a different scaling factor to make sure NN inputs stay within distribution.
        curr_scaling_mm = self.mag_model_config['input_std']*0.001
        new_scaling = mag_thres/np.sqrt(2)
        magout = np.zeros((len(contact_location_xy),3))
        scaled_loc = torch.tensor(contact_location_xy/new_scaling, dtype=torch.float)

        with torch.no_grad():
            magdist = self.mag_model(scaled_loc)
            magdist = magdist[mags_in_range]
        magout[mags_in_range] = torch.distributions.normal.Normal(
            magdist[...,:3], torch.exp(magdist[...,3:])).sample()
        # magout[mags_in_range] = magdist[...,:3]

        return magout

class FMBase(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'pose_err'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": -1.0,
        "bonus": 4.0,
        "penalty": -50,
    }
    def __init__(self, model_path, **kwargs):

        curr_dir = os.path.dirname(os.path.abspath(__file__))

        # Process model to ues DManus as end effector
        raw_sim = env_base.get_sim(model_path=curr_dir+model_path)
        raw_xml = raw_sim.model.get_xml()
        processed_xml = reassign_parent(xml_str=raw_xml, receiver_node="panda0_link7", donor_node="DManus_mount")
        processed_model_path = curr_dir+model_path[:-4]+"_processed.xml"
        with open(processed_model_path, 'w') as file:
            file.write(processed_xml)

        # self.sim = env_base.get_sim(model_path=processed_model_path)
        # self.sim_obsd = env_base.get_sim(model_path=processed_model_path)
        # os.remove(processed_model_path)
        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=processed_model_path)
        os.remove(processed_model_path)

        self._setup(**kwargs)


    def _setup(self, 
               target_pose, 
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs):

        self.target_pose = target_pose

        # get env
        # env_base.MujocoEnv.__init__(self,
        #                         # sim = self.sim,
        #                         # sim_obsd = self.sim_obsd,
        #                         frame_skip = 40,
        #                         config_path = config_path,
        #                         obs_keys = self.DEFAULT_OBS_KEYS,
        #                         rwd_keys_wt = self.DEFAULT_RWD_KEYS_AND_WEIGHTS,
        #                         rwd_mode = "dense",
        #                         act_mode = "pos",
        #                         act_normalized = True,
        #                         is_hardware = False,
        #                         **kwargs)
        
        env_base.MujocoEnv.__init__(self,
                                processed_model_path)
        
        os.remove(processed_model_path)

        env_base.MujocoEnv._setup(self, 
            obs_keys=self.DEFAULT_OBS_KEYS,
            weighted_reward_keys = self.DEFAULT_RWD_KEYS_AND_WEIGHTS,
            reward_mode = "dense",             
            frame_skip = 40,
            normalize_act = True,
            # obs_range = (-10, 10),
            # seed = None,
            act_mode = "pos",
            is_hardware = False,
            config_path = config_path,
            # rwd_viz = False,
            robot_name = 'fm', 
        )


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
        obs_dict['pose_err'] = obs_dict['qp'] - self.target_pose
        return obs_dict


    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict['pose_err'], axis=-1)
        far_th = 10

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   reach_dist),
            ('bonus',   (reach_dist<1) + (reach_dist<2)),
            ('penalty', (reach_dist>far_th)),
            # Must keys
            ('sparse',  -1.0*reach_dist),
            ('solved',  reach_dist<.5),
            ('done',    reach_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

class DManusEnvFixed(DManusBase):

    def reset(self):
        obs = super().reset(self.init_qpos, self.init_qvel)
        return obs

class DManusEnvWrist(DManusBase):
    def reset(self):
        obs = super().reset(self.init_qpos, self.init_qvel)
        return obs

class FMReachEnvFixed(FMBase):

    def reset(self):
        obs = super().reset(self.init_qpos, self.init_qvel)
        return obs


class FMReachEnvRandom(FMBase):

    def reset(self):
        self.target_pose = self.np_random.uniform(low=self.sim.model.actuator_ctrlrange[:,0], high=self.sim.model.actuator_ctrlrange[:,1])
        obs = super().reset(self.init_qpos, self.init_qvel)
        return obs
