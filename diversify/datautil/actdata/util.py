# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch
from torchvision import transforms
import numpy as np

import datasets


def act_train():
    return transforms.Compose([
        transforms.ToTensor()
    ])


def loaddata_from_numpy(dataset='dsads', task='cross_people', root_dir='./data/act/'):
    if dataset == 'pamap' and task == 'cross_people':
        x = np.load(root_dir+dataset+'/'+dataset+'_x1.npy')
        ty = np.load(root_dir+dataset+'/'+dataset+'_y1.npy')
        cy, py, sy = ty[:, 0], ty[:, 1], ty[:, 2]

    elif dataset == datasets.SpeechCommand:
        x = torch.cat([
            torch.load(root_dir + dataset + '/train_a.pt'),
            torch.load(root_dir + dataset + '/train_b.pt'),
            torch.load(root_dir + dataset + '/train_c.pt'),
            torch.load(root_dir + dataset + '/train_d.pt'),
        ])
        y = torch.load(root_dir + dataset + '/train_y.pt') 
        cy = y
        py, sy = None, None  

    else:
        x = np.load(root_dir+dataset+'/'+dataset+'_x.npy')
        ty = np.load(root_dir+dataset+'/'+dataset+'_y.npy')
        cy, py, sy = ty[:, 0], ty[:, 1], ty[:, 2]

    return x, cy, py, sy
