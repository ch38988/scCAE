#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 2 15:14:11 2022

@author: ch38988
"""

import math
import torch
from torch import nn


class AE(torch.nn.Module): 
    """#autoencoder"""
    def __init__(self, D_in=None, latent_dim=None, drop_rate=0.5):
        super(AE,self).__init__()
        self.encoder = torch.nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,latent_dim)
            )
        self.decoder = torch.nn.Sequential(
            nn.Linear(latent_dim,256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, D_in),
            nn.Sigmoid()  #output scaled to (0,1)
            )
        
    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1,h2


class CAE(torch.nn.Module):
    """#contractive autoencoder"""
    def __init__(self, D_in=None, latent_dim=None, drop_rate=0.5):
        super(CAE,self).__init__()
        self.encoder = torch.nn.Sequential(
            nn.Dropout(drop_rate), #dropout nodes to avoid overfitting
            nn.Linear(D_in, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,latent_dim),
            nn.ReLU()
            )
        self.decoder = torch.nn.Sequential(
            nn.Linear(latent_dim,256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, D_in),
            nn.Sigmoid()  #output scaled to (0,1)
            )
        
    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1,h2
