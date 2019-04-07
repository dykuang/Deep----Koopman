# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:27:38 2019

@author: dykua

Network architecture
"""

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, LSTM, Input, Lambda, TimeDistributed, Conv1D, Flatten,Concatenate, Reshape, Add
#from keras import regularizers
import tensorflow as tf
import keras.backend as KB
from keras.regularizers import l1
from keras.engine.topology import Layer
from keras.initializers import RandomNormal, Zeros

def _transformer(x, hid_dim, num_layers, out_dim, inter_dim_list=[32, 32], activation_out = 'tanh'):
    if num_layers:
        for i in range(num_layers):
            x = LSTM(hid_dim,return_sequences=True)(x)
#    x_r = TimeDistributed(Dense(out_dim, activation='tanh'))(x)
#    x = TimeDistributed(Dense(out_dim))(x)
    for j in inter_dim_list:
        x = TimeDistributed(Dense(j, activation='tanh'))(x)
    x = TimeDistributed(Dense(out_dim, activation=activation_out))(x)
#    x = Add()([x_r, x_a])
    return x

def pred_K(x_in, num_complex, num_real,  K_reg):
    xk1 = Conv1D(8, kernel_size=2, strides=1, padding='same', activation='relu')(x_in) # (samples, steps, channel=1)
    xk2 = Conv1D(8, kernel_size=3, strides=1, padding = 'same', activation='relu')(x_in)
    xk = Concatenate()([xk1, xk2])
    xk = Conv1D(16, kernel_size = 3, strides=2, padding = 'valid')(xk)
    xk = Reshape([-1])(xk)
    #xk = Dense(16)(xk)
    Koop = Dense(num_complex*2 + num_real, activity_regularizer = l1(K_reg), activation='linear', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-3)
)(xk)
    
    return Koop

class linear_update(Layer):

    def __init__(self, output_dim, num_complex, num_real, **kwargs):
        self.output_dim = output_dim
        self.kernels = []
        self.num_complex = num_complex
        self.num_real = num_real
        
        super(linear_update, self).__init__(**kwargs)

    def build(self, input_shape):
#        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        super(linear_update, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        y, Km = x
        C_seq = []
        R_seq = []
        # forming complex block : batchsize, 2, 2
        if self.num_complex:
            
            for count_c in range(self.num_complex):
                scale = tf.exp(Km[:, count_c])
#                scale = Km[:, count_c]
                cs = tf.cos(Km[:, count_c + self.num_complex])
                sn = tf.sin(Km[:, count_c + self.num_complex])
                real = tf.multiply(scale, cs)
                img = tf.multiply(scale, sn)
    #            print(KB.int_shape(real))
                block = tf.stack([real, -img, img, real], axis = 1)
                Ci = tf.reshape(block, (-1, 2, 2))
                C_seq_i = []
                C_seq_i.append(y[:,0,(2*count_c):(2*count_c+2)]) 
                for i in range(self.output_dim[0]-1):
                    C_seq_i.append(tf.einsum('ik,ikj->ij', C_seq_i[i], Ci))
                C_seq.append(tf.stack(C_seq_i, axis=1))
#            print(KB.int_shape(block))   
            C_seq_tensor = tf.reshape(tf.stack(C_seq, axis = 2), (-1, self.output_dim[0], 2*self.num_complex))
            print(C_seq_tensor.shape)
#        print(KB.int_shape(tf.stack(C_seq_i, axis = 1)))        
        # forming real block: batchsize, 1
        if self.num_real:
            R = tf.exp(Km[:,(2*self.num_complex):])
#            R = Km[:,(2*par['num complex']):]
            R_seq.append(y[:,0, (2*self.num_complex):]) 
            for i in range(self.output_dim[0]-1):
                R_seq.append(tf.multiply(R_seq[i], R))
            
        
            R_seq_tensor = tf.stack(R_seq, axis = 1)
            print(R_seq_tensor.shape)
#        print(tf.concat([C_seq_tensor, R_seq_tensor], axis=2).shape)

        if self.num_complex and self.num_real:
            return tf.concat([C_seq_tensor, R_seq_tensor], axis=2)
        
        elif self.num_real:
            return R_seq_tensor
        elif self.num_complex:
            return C_seq_tensor
        
        
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_y, shape_Km = input_shape
        return (shape_y[0], self.output_dim[0], self.output_dim[1])
    

def Build_Model(params):
    input_shape = (params['pred steps'], params['feature dim'])
    x_in = Input(input_shape)
    Gx = _transformer(x_in, params['hidden dim'], params['en layer'], params['latent dim'], params['en dim list'])
    
    Koop = pred_K(Gx, params['num complex'], params['num real'], params['K reg']) # get prediction for the matrix for update
    LU = linear_update(output_dim = (params['pred steps'], params['latent dim']), num_complex = params ['num complex'], num_real = params ['num real'])
    KGx = LU([Gx, Koop])
    
    decoder_input = Input(shape = (params['pred steps'], params['latent dim']))
    decoded = _transformer(decoder_input, params['hidden dim'], params['de layer'], params['feature dim'], params['de dim list'], 'linear')
    _decoder = Model(decoder_input, decoded)
    
    encoder = Model(x_in, Gx)
    Knet = Model(x_in, [Koop, KGx])
    
    return [encoder, Knet, _decoder]