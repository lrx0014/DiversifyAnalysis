# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import datasets
from diversify.datautil.actdata.util import *
from diversify.datautil.util import mydataset, Nmax
import numpy as np
import torch


class ActList(mydataset):
    def __init__(self, args, dataset, root_dir, people_group, group_num, transform=None, target_transform=None, pclabels=None, pdlabels=None, shuffle_grid=True):
        super(ActList, self).__init__(args)
        self.domain_num = 0
        self.dataset = dataset
        self.task = 'cross_people'
        self.transform = transform
        self.target_transform = target_transform
        x, cy, py, sy = loaddata_from_numpy(self.dataset, self.task, root_dir)
        self.people_group = people_group
        self.position = np.sort(np.unique(sy))
        if self.dataset == datasets.EMG:
            self.comb_position(x, cy, py, sy)
        if self.dataset == datasets.SpeechCommand:
            self.x, self.labels = x, cy
        self.x = self.x[:, :, np.newaxis, :]
        self.transform = None
        self.x = torch.tensor(self.x).float()
        if pclabels is not None:
            self.pclabels = pclabels
        else:
            self.pclabels = np.ones(self.labels.shape)*(-1)
        if pdlabels is not None:
            self.pdlabels = pdlabels
        else:
            self.pdlabels = np.ones(self.labels.shape)*(0)
        self.tdlabels = np.ones(self.labels.shape)*group_num
        self.dlabels = np.ones(self.labels.shape) * \
            (group_num-Nmax(args, group_num))

    def comb_position(self, x, cy, py, sy):
        for i, peo in enumerate(self.people_group):
            index = np.where(py == peo)[0]
            tx, tcy, tsy = x[index], cy[index], sy[index]
            for j, sen in enumerate(self.position):
                index = np.where(tsy == sen)[0]
                if j == 0:
                    ttx, ttcy = tx[index], tcy[index]
                else:
                    ttx = np.hstack((ttx, tx[index]))
            if i == 0:
                self.x, self.labels = ttx, ttcy
            else:
                self.x, self.labels = np.vstack(
                    (self.x, ttx)), np.hstack((self.labels, ttcy))

    # def comb_position(self, x, py, cy=None, sy=None):

    #     for i, peo in enumerate(self.people_group):
    #         index = np.where(py == peo)[0]
            
    #         tx = x[index]
    #         tcy = cy[index] if cy is not None else py[index]  
    #         tsy = sy[index] if sy is not None else None 
            
    #         if tsy is not None:
    #             for j, sen in enumerate(self.position):
    #                 index = np.where(tsy == sen)[0]
    #                 if j == 0:
    #                     ttx, ttcy = tx[index], tcy[index]
    #                 else:
    #                     ttx = np.hstack((ttx, tx[index]))
    #         else:
    #             ttx, ttcy = tx, tcy

    #         if i == 0:
    #             self.x, self.labels = ttx, ttcy
    #         else:
    #             self.x = np.vstack((self.x, ttx))
    #             self.labels = np.hstack((self.labels, ttcy))


    def set_x(self, x):
        self.x = x
