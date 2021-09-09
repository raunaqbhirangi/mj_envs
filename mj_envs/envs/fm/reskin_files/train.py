import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data_handling import MagneticSensorData

if __name__ == '__main__':
    data = MagneticSensorData(['./data'],2)
    
    n_epochs = 20
    batch_size = 128

    label_mean, label_std = 0.,1.
    input_mean, input_std = 0.,1.

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
    ])
    
    trainLoader = DataLoader()
    model = torch.Sequential()
    opt = optim.Adam(lr=1e-3)

    for e in range(n_epochs):
        for i,sample in enumerate(trainLoader):
            opt.zero_grad()
            
            loc, sens, force = sample.values()
            # Transform sample
            force_locs = torch.repeat_interleave(loc,5,dim=0) - mag_locations.repeat(batch_size,1)
            label_mags = torch.matmul(
                mag_tfs.repeat(batch_size,1,1),
                sens.reshape((-1,3,1)))

            pred_mags = model(force_locs)
            
            # Compute Loss
            loss = pred_mags - label_mags
            loss.backward()
            opt.step()