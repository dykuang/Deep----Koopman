# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:01:04 2019

@author: dykua

a script getting information from trained model

"""

import numpy as np
from scipy.io import loadmat
from keras.models import load_model, Model
from Architecture import Build_Model
from keras.layers import Input
import pickle
exp_name = 'two_basins'

with open("params_{}.pkl".format(exp_name), "rb") as file:
    params = pickle.load(file)

#params = np.load('parames_{}.npy'.format(exp_name)).item()

data = loadmat(params['data name'])['X']

#data = data[::4]

def create_dataset1(dataset, look_back=params['pred steps']):
    dataX, dataY = [], []
    for j in range(len(dataset)):
        a = dataset[j,:look_back, :]
        dataX.append(a)
        dataY.append(dataset[j,1 : look_back+1, :])
    return np.array(dataX), np.array(dataY)

dataX, dataY = create_dataset1(data)

model_seq = Build_Model(params)
model_name=['encoder', 'Knet', 'decoder']
for i in range(len(model_seq)):
    model_seq[i].load_weights('DK_{}_{}.h5'.format(params['save name'], model_name[i]))
    

encoder, Knet, decoder = model_seq



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# visualization

# latent dynamics vs latent linear updated dynamics
x_l = encoder.predict(dataX)
plt.figure()
for i in range(params['num_samples']):
    plt.plot(x_l[i,:,0], x_l[i,:,1])
plt.title('First two dimensions in latent')

# lambda and omega of the learned eigenvalues
   
Koos, x_kl = Knet.predict(dataX)

plt.figure()
for i in range(params['num_samples']):
    plt.plot(x_kl[i,:,0], x_kl[i,:,1])
plt.title('First two dimensions in latent: linearly approximated')
    
# if in 3d:
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#for i in range(params['num_samples']):
#    ax.plot(x_kl[i,:,0], x_kl[i,:,1], x_kl[i,:,2])
    
#fig = plt.figure()
##for i in range(0, X_stacked.shape[1], 2):
#plt.plot(dataX[:,0,1],Koos[:,0],'-*')

# reconstructed dynamics vs reconstructed dynmaics from linear approximation
x_r = decoder.predict(x_l)
x_lr = decoder.predict(x_kl)
plt.figure()
for i in range(params['num_samples']):
    plt.plot(x_r[i,:,0], x_r[i,:,1])
plt.title('First two dim: reconstruction')

plt.figure()
for i in range(params['num_samples']):
    plt.plot(x_lr[i,:,0], x_lr[i,:,1])
plt.title('First two dim: reconstruction from linear update')
    
x_kl_reshaped = x_kl[::2].reshape((-1, params['latent dim']))
#x_l_reshaped = x_l[::10].reshape((-1, params['latent dim']))
points = dataX[::2].reshape((-1,2))
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(points[:,0], points[:,1], x_kl_reshaped[:,1])
#plt.figure()
#plt.scatter(points[:,0], points[:,1], c=x_kl_reshaped[:,1], cmap='bwr')


# plot eigenfunction
x_min = np.min(dataX[...,0])
x_max = np.max(dataX[...,0])
y_min = np.min(dataX[...,1])
y_max = np.max(dataX[...,1])

res_x = params['pred steps'] # problem when >2? may because of the training set.
res_y = params['pred steps']

x = np.linspace(x_min, x_max, res_x)
y = np.linspace(y_min, y_max, res_y)
grid_x, grid_y = np.meshgrid(x, y)
grid = np.zeros((res_x*res_y, 2))
grid[:,0] = grid_x.flatten()
grid[:,1] = grid_y.flatten()

grid = np.tile(grid, params['pred steps'])

grid_reshaped = np.reshape(grid, (-1, params['pred steps'], params['feature dim']))

g_val = encoder.predict(grid_reshaped)[:,-1,:]

#g_val = np.median(g_val, axis = 1)

Eig, g_val_1 = Knet.predict(grid_reshaped)

gn = np.sum(np.square(g_val), axis = 1)**0.5

#g_val_r = np.reshape(g_val, (-1, params['latent dim']))

#g_x = np.reshape(g_val_1[...,0], (res_x,res_y))
#g_y = np.reshape(g_val_1[...,1], (res_x,res_y))

#g_n = g_x**2 + g_y**2

#plt.figure()
#plt.imshow(g_x[::-1,:])
#plt.contour(g_x[::-1,:], linewidths=2,  colors=['black'])
#
#plt.figure()
#plt.imshow(g_y[::-1,:])
#plt.contour(g_y[::-1,:], linewidths=2,  colors=['black'])
#
#plt.figure()
#plt.imshow(g_n[::-1,:])
#plt.contour(g_n[::-1,:], linewidths=2,  colors=['black'])



def vis_eigfunc(val):
    grid_reshaped = val.reshape(res_x,res_y)

    plt.figure()
    plt.imshow(grid_reshaped.transpose(), cmap='bwr')
#    plt.xticks(ticks=np.arange(0,res_x,res_x/10), labels=np.round(np.arange(x_min, x_max, (x_max-x_min)/10),2))
#    plt.yticks(ticks=np.arange(res_y,0,-res_y/10), labels=np.round(np.arange(y_min, y_max, (y_max-y_min)/10),2))
    
vis_eigfunc(gn)