# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:47:34 2019

@author: dykua

train a model
"""

from keras.models import Model
import tensorflow as tf
from keras.layers import Lambda
# setting parameters
par = {'loss_weights': [1.0, 1.0],  # state, reconstruction
       'pred steps': 40,
       'batch size': 32,
       'hidden dim': 16,
#       'latent dim': 5,
       'en layer': 0,
       'de layer': 0,
       'en dim list': [64, 32],
       'de dim list': [64, 32],
       'K reg': 1e-4,
       'epochs': 1000,
       'data name': 'two_basins.mat',
       'num complex': 4,  # number of conjugate pairs
       'num real': 1,     # number of real eigenvalues
       'lr': 5e-4,
       'save name': 'two_basins'
       }

par['latent dim'] = 2*par['num complex'] + par['num real']

#prepare data
from scipy.io import loadmat
from Utils import create_dataset1
dataset=loadmat(par['data name'])['X']
print(dataset.shape)

dataset=dataset[::2]
look_back = par['pred steps']
par['num_samples'], par['time steps'], par['feature dim'] = dataset.shape
par['num per track'] = par['time steps'] - par['pred steps'] + 1

# fix random seed for reproducibility
import numpy as np
np.random.seed(7)
trainX, trainY = create_dataset1(dataset, look_back)

'''
Encoder part
'''
from keras.layers import Input
input_shape = (par['pred steps'], par['feature dim'])
x_in = Input(input_shape)

from Architecture import _transformer, pred_K, linear_update

Gx = _transformer(x_in, par['hidden dim'], par['en layer'], par['latent dim'], par['en dim list'])

'''
linear update in latent space
'''
Koop = pred_K(Gx, par['num complex'], par['num real'], par['K reg']) # get prediction for the matrix for update
LU = linear_update(output_dim = (par['pred steps'], par['latent dim']), num_complex = par['num complex'], num_real = par['num real'])
KGx = LU([Gx, Koop])


'''
Decoder part
'''
decoder_input = Input(shape = (par['pred steps'], par['latent dim']))
decoded = _transformer(decoder_input, par['hidden dim'], par['de layer'], par['feature dim'], par['de dim list'], 'linear')
_decoder = Model(decoder_input, decoded)
decoded_x = _decoder(Gx)
decoded_xp = _decoder(KGx)

'''
Losses
'''
from keras.losses import mean_squared_error, mean_absolute_error

def S_error(args):
    Y0, Y1 = args
    return tf.reduce_mean(tf.abs(Y0-Y1))
#    return mean_squared_error(Y0, Y1)

state_err = Lambda(S_error)([Gx, KGx])

def State_loss(yTrue, yPred):
    return tf.reduce_mean(tf.abs(Gx - KGx))
#    return mean_squared_error(Gx, KGx)

rec_err = Lambda(S_error)([x_in, decoded_x])

def Rec_loss(yTrue, yPred):
    return tf.reduce_mean(tf.abs(x_in-decoded_x))
#    return mean_squared_error(x_in, decoded_x)

'''
Models
'''
encoder = Model(x_in, Gx)
Knet = Model(x_in, [Koop, KGx])
full_model = Model(x_in, decoded_xp)
print(full_model.summary())


def customLoss(weights = par['loss_weights']):
    def Loss(yTrue, yPred):
        return mean_absolute_error(yTrue, yPred) \
                + par['loss_weights'][0]*state_err \
                + par['loss_weights'][1]*rec_err
#                + par['loss_weights'][2]*mean_absolute_error(K, 0) 
    
    return Loss

'''
training
'''        
from keras.optimizers import Adam
full_model.compile(loss=customLoss(), metrics=[State_loss, Rec_loss],
                   optimizer=Adam(lr = par['lr'], decay = 1e-4))

history = full_model.fit(trainX, trainX, epochs=par['epochs'], batch_size=par['batch size'], verbose=1)

par['training history'] = history.history
'''
Check trained models and save
'''

# training loss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.figure()
plt.plot(np.log(history.history['loss']), 'r')
plt.plot(np.log(history.history['State_loss']),'k')
plt.plot(np.log(history.history['Rec_loss']), 'b')
plt.legend(['loss', 'State_loss', 'Rec_loss'])

print(history.history['State_loss'][::100])

#Koos,_ = Knet.predict(trainX)
#    
#plt.figure()
#for i in range(dataset.shape[0]):
#    plt.plot(dataset[i,:,0], dataset[i,:,1])
#plt.legend([i for i in range(trainX.shape[0])])
#
#pred = full_model.predict(trainX)
#plt.figure()
#for i in range(pred.shape[0]):
#    plt.plot(pred[i,:,0], pred[i,:,1])
#plt.title('reconstruction')
#
#plt.figure()
#for i in range(trainX.shape[0]):
#    plt.plot(trainX[i,:,0], trainX[i,:,1])
#plt.title('original')   



#plt.figure()
#plt.plot(dataset[:,0,1], Koos[:,0], '-*')
#plt.xlabel('$y_0$')
#plt.ylabel('$\lambda$')
#plt.title('Exp__{}'.format(par['save name']))


# save model and parameters
model_seq=[encoder, Knet, _decoder]
model_name=['encoder', 'Knet', 'decoder']
for i in range(len(model_seq)):
    model_seq[i].save_weights('DK_{}_{}.h5'.format(par['save name'], model_name[i]))

import pickle

with open("params_{}.pkl".format(par['save name']), "wb") as file:
    pickle.dump(par, file, protocol=pickle.HIGHEST_PROTOCOL)
#np.save('parames_{}.npy'.format(par['save name']), par) # could use pickle
