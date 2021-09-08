import numpy as np
import torch
from torch.utils.data import DataLoader
from data_handling import MagneticSensorData

if __name__ == '__main__':
    data = MagneticSensorData(['./data'],2)
    
    n_epochs = 20
    mag_locations = torch.tensor([
        [8.,8.],
        [8.,15.],
        [1.,8.],
        [8.,1.],
        [15.,8.],
    ])
    # Based on Tess visualizer
    mag_tfs = torch.tensor([
        [[1.,0.,0.],
         [0.,1.,0.],
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
        [[-1.,0.,0.],
         [0.,-1.,0.],
         [0.,0.,1.]],
    ])
    
    trainLoader = DataLoader()
    for e in range(n_epochs):
        # Get sample

        # Transform sample

        # Compute Loss

        # opt.step()