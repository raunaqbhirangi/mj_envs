import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from .data_handling import MagneticSensorData
except:
    from data_handling import MagneticSensorData

import ipdb

def GaussianNLLLoss(input_t:torch.tensor, target_t:torch.tensor, var_t:torch.tensor, eps:float=1e-6):
    if not torch.is_tensor(eps):
        eps = torch.tensor(eps)
    clamped_var_t = torch.max(var_t,eps.expand_as(var_t))
    loss = 0.5 * (torch.sum(
        torch.log(clamped_var_t) + torch.square(input_t - target_t)/clamped_var_t,
        axis=-1))
    # ipdb.set_trace()    
    return torch.mean(loss)
    
class ReSkinSim(nn.Module):
    def __init__(self, n_input=2, n_output=3):
        super(ReSkinSim, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,2*n_output)
        )

    def forward(self, input_t):
        return self.model(input_t)
    
if __name__ == '__main__':
    data = MagneticSensorData(['./data'],2,scale_std=1)
    
    n_epochs = 20
    batch_size = 128

    # Input is relative location, output is magnetic field
    n_input = 2
    n_output = 3

    input_mean, input_std = 0.,8.

    
    mag_locations = torch.tensor([
        [8.,8.],
        [1.,8.],
        [8.,1.],
        [15.,8.],
        [8.,15.],
    ])
    # Based on Tess visualizer
    mag_tfs = torch.tensor([
        [[1.,0.,0.],
         [0.,1.,0.],
         [0.,0.,1.]],
        [[-1.,0.,0.],
         [0.,-1.,0.],
         [0.,0.,1.]],
        [[1.,0.,0.],
         [0.,1.,0.],
         [0.,0.,1.]],
        [[0.,-1.,0.],
         [1.,0.,0.],
         [0.,0.,1.]],
        [[0.,1.,0.],
         [-1.,0.,0.],
         [0.,0.,1.]],
    ],  dtype=torch.float)
    
    data_mag_std = torch.from_numpy(data.sensor_std).float()
    mag_std_raw = torch.tensor(data.sensor_std.reshape((5,3,1)), dtype=torch.float)
    mag_std_tf = torch.abs(torch.matmul(mag_tfs, mag_std_raw))
    mag_std = torch.sqrt(torch.mean(torch.square(torch.squeeze(mag_std_tf)),axis=0))
    mag_std_scale = 3.

    config = {
        'input_mean':input_mean,
        'input_std': input_std,
        'mag_std': mag_std.tolist(),
        'mag_std_scale': mag_std_scale,
    }
    # ipdb.set_trace()
    trainLoader = DataLoader(data, batch_size = batch_size, shuffle=True)
    
    model = ReSkinSim()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for e in range(n_epochs):
        total_loss = 0.
        for i,sample in enumerate(trainLoader):
            opt.zero_grad()
            
            loc, sens, force = sample.values()
            
            sens_raw = (sens.float() * data_mag_std)
            # Get five datapoints from each sample, and rotate magnetometer 
            # readings appropriately
            force_locs = torch.repeat_interleave(loc.float(),5,dim=0)[...,:2] - mag_locations.repeat(loc.shape[0],1)
            force_locs_norm = (force_locs - input_mean)/input_std

            label_mags = torch.matmul(
                mag_tfs.repeat(sens.shape[0],1,1),
                sens_raw.reshape((-1,3,1)))

            label_mags = torch.squeeze(label_mags/(mag_std_scale*mag_std.view(1,3,1)))
            # ipdb.set_trace()
            pred_mags = model(force_locs)
            
            # Compute Loss
            loss = GaussianNLLLoss(
                pred_mags[...,:3], label_mags, torch.exp(pred_mags[...,3:]))
            loss.backward()
            opt.step()

            total_loss += loss.item()
        print('Epoch: {}: Loss: {:.2e}'.format(e+1, total_loss))
    
    torch.save(model.state_dict(), './reskin_sim_weights')
    with open('./model_config.yaml','w') as f:
        yaml.dump(config, f)
    
    # Test loop
    test_data = data[:10]
    loc, sens, force = test_data.values()
    loc = torch.from_numpy(loc)
    sens = torch.from_numpy(sens)
    force = torch.from_numpy(force)

    sens_raw = (sens.float() * data_mag_std)
    label_mags = torch.matmul(
                mag_tfs.repeat(sens.shape[0],1,1),
                sens_raw.reshape((-1,3,1))).squeeze()

    force_locs = torch.repeat_interleave(loc.float(),5,dim=0)[...,:2] - mag_locations.repeat(loc.shape[0],1)
    force_locs_norm = (force_locs - input_mean)/input_std

    pred_mags = model(force_locs)
    print('Predictions:')
    print(pred_mags[...,:3] * mag_std.view(1,3) - label_mags)
    print(torch.exp(pred_mags[...,3:]) * mag_std.view(1,3)*mag_std_scale)
