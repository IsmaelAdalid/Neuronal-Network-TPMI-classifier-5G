import time
import sys
from sklearn.utils import shuffle

import tensorflow
from tensorflow import keras
import os
import numpy as np

from keras.layers import Layer, Conv2D, LayerNormalization, ConvLSTM2D, ConvLSTM3D, \
    TimeDistributed, Conv3D, InputLayer, Bidirectional, GRU, Flatten, \
    Reshape, MultiHeadAttention, Conv1D, Dropout,Concatenate,BatchNormalization,Dense,Softmax
from tensorflow._api.v2.nn import relu
from keras import Model
from keras.utils import Sequence
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot
import dill as pickle
import pandas as pd
import random
@keras.utils.register_keras_serializable(package="Custom")
class Multinput_classifier(Layer):

    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)
        self._params = params
        self.conv = []
        self.batch = []
        self.fully = []
        self.drop = []

    def build(self, input_shape):

        # Normalizamod la entrada
        self.input_norm = BatchNormalization(axis=[-1, -2, -3])

        # Input convolution
        self.input_conv = Conv2D(filters=self._params['n_filters_1'],
                                  kernel_size=self._params['kernel_size_input_1'],
                                  padding='same',
                                  activation=None,
                                 input_shape=input_shape)

        # self.input_2layer = InputLayer(input_shape=(2,1)) # Entrada pathloss y ruido

        self.input_2layer = InputLayer()  # Entrada pathloss y ruido
        # Bloques layers

        for i in range(1, len(self._params['n_filters_inside'])):
            self.conv.append( Conv2D(filters=self._params['n_filters_inside'][i],
                                  kernel_size=self._params['kernel_inside'][i],
                                  padding='valid', strides=self._params['strides_inside'][i],
                                  activation='relu'))
            self.batch.append( BatchNormalization(axis=[-1, -2, -3] ))

        #Bloques Fully Connected
        for i in self._params['n_nodes_dense']:
            self.fully.append(Dense(units=i, activation='relu'))
            self.drop.append(Dropout(rate=self._params['Dropout_Prob']))

        # Bloque de activacioon
        self.clasification = Dense(units=6,activation='softmax')

        # Normalization layer
        self.layer_normalization = LayerNormalization(axis = -1)

        self.concat = Concatenate(axis=-1)




    def call(self, input):

        # inp_conv = input[:,:,:,0]
        inp_conv = tf.expand_dims(input[:,:,0:8],axis=-1)
        inp_esc = tf.math.real(input[:,0,8:10])

        # inp_conv = tf.expand_dims(input['conv'],axis=-1)
        # inp_esc = tf.squeeze(input['esc'],axis=1)

        # Primera parte conv
        z = tf.concat([tf.math.real(inp_conv), tf.math.imag(inp_conv)], axis=-1)

        z = self.input_conv(z)
        z = self.input_norm(z)

        for c, bn in zip(self.conv, self.batch):
            z = c(z)
            z = bn(z)

        z_flatten = Flatten(data_format="channels_last")(z)

        # Primera parte escalar
        e = self.input_2layer(inp_esc)
        e = self.layer_normalization(e)

        # Parte común

        w = self.concat([e, z_flatten])




        for f, d in zip(self.fully, self.drop):
            w = f(w)
            w = d(w)

        # Return softmax probabilities
        return self.clasification(w)

    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0:-2] +  (self._params['n_filters'],))
class ModelMultiinput_Classifier(Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, params,**kwargs):
        super().__init__(**kwargs)
        self.myLayers = Multinput_classifier(params)

    def call(self, inputs):
        return self.myLayers(inputs)



def predice(fil,x):
    with open(fil+"modelTPM_potenteI.pkl",'rb') as fl:
        model = pickle.load(fl)
    y = np.argmax(model.predict(x),axis=1)
    return y,model

fil =    # Put the path where agent is saved
input_predice = np.zeros(1,120,10)
input_predice[:,0:8]  = # allocate the channel
input_predice[:,9] = # Allocate the noise estimation
input_predice[:,10] # Allocate the pathloss

y,model = predice(fil,input_predice)
print(y)
