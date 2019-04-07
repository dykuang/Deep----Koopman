# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:21:29 2019

@author: dykua

some utility functions
"""

import numpy as np

def create_dataset(dataset, look_back, total_time_steps):
    dataX, dataY = [], []
    for j in range(len(dataset)):
        for i in range(total_time_steps-look_back-1):
		       a = dataset[j,i:(i+look_back), :]
		       dataX.append(a)
		       dataY.append(dataset[j,(i+1):(i + look_back+1), :])
    return np.array(dataX), np.array(dataY)

def create_dataset1(dataset, look_back):
    dataX, dataY = [], []
    for j in range(len(dataset)):
        a = dataset[j,:look_back, :]
        dataX.append(a)
        dataY.append(dataset[j,1 : look_back+1, :])
    return np.array(dataX), np.array(dataY)

def normalize(X):
    # transforms per training tracjatory into a unit cube, X: track, time, dim
    for num in range(X.shape[0]):
        for dim in range(X.shape[2]):
            _max = np.max(X[:,dim])
            _min = np.min(X[:,dim])
            X[num,:,dim] = np.divide(X[num,:,dim]-_min, _max - _min)
    return X



