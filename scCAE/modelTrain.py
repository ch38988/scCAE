#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 1 20:55:12 2022

@author: ch38988
"""

import os
import argparse
import math
import time
import random
import pandas as pd
import numpy as np
import scipy as sp

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import sklearn.metrics as skm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

from model import *


#gpu or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def loss_function(W, x, recons_x, h, lam):
    """Compute the Contractive AutoEncoder Loss
    Evalutes the CAE loss, which is composed as the summation of a Mean
    Squared Error and the weighted l2-norm of the Jacobian of the hidden
    units with respect to the inputs.
        `W` (FloatTensor): (N_hidden x N), where N_hidden and N are the
          dimensions of the hidden units and input respectively.
        `x` (Variable): the input to the network, with dims (N_batch x N)
        recons_x (Variable): the reconstruction of the input, with dims
          N_batch x N.
        `h` (Variable): the hidden units of the network, with dims
          batch_size x N_hidden
        `lam` (float): the weight given to the jacobian regulariser term
    Returns:
        Variable: the (scalar) CAE loss
    """
    criterion = nn.MSELoss()
    mse = criterion(recons_x,x)
    # W is shape of N_hidden x N
    dh = h * (1 - h) # Hadamard product produces size N_batch x N_hidden
    # Sum through the input dimension to improve efficiency, as suggested in #1
    w_sum = torch.sum(Variable(W)**2, dim=1)
    # unsqueeze to avoid issues with torch.mv
    w_sum = w_sum.unsqueeze(1) # shape N_hidden x 1
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)





####################main##########################
def main():
    """ Main method """
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--input_file', type=str,
                        default="/home/che82/project/psychAD/h5ad/bydonor/M10031.csv",
                        help='input RNAseq dataset')
    
    args = parser.parse_args()
    print(args)
    
    if os.path.exists(args.input_file):
        """Step 1: data reading and preprocessing"""
        print('Reading files...')
        data0 = pd.read_csv(args.input_file,index_col=0)
        obs = torch.from_numpy(preprocessing.StandardScaler().fit_transform(data0.values).astype('float64'))
        
        """Step 2: create the model"""
        print('Defining the DL model...')
        loader = DataLoader(dataset=obs,batch_size=100,shuffle=True)
        model = CAE(D_in=obs.shape[1],latent_dim=2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)
    
        """Step 3: start training"""
        print('training start...')
        epochs = 10
        losses = []
        for epoch in range(epochs):
            print("\n*** Epoch = %d ***"%(epoch))
            tmp_losses = []
            for bc in loader: #train and loss in batch
                dat = bc.float().to(device)
                emb, recon = model(dat)
                W = model.state_dict()['encoder.9.weight']
                loss = loss_function(W, dat, recon,emb, lam=0.5)
                dat.requires_grad_(False)
                tmp_loss = loss.item()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tmp_losses.append(tmp_loss)
            losses.append(np.mean(tmp_losses))
            print("Loss = %.4f"%np.mean(tmp_losses))  
        print('Training model completed, ready for some downstream analysis')
    else:
        print('Check the file existance!!!')
                                                                                        

if __name__ == '__main__':
    main()





