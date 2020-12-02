# -*- coding: utf-8 -*-
"""
COMP 5600/6600/6606 FINAL PROJECT

TEAM: 8-BIT BRAIN

@author: Dennis Brown (dgb0028@auburn.edu)
@author: Shannon McDade (slm0035@auburn.edu)
@author: Jacob Parmer (jdp0061@auburn.edu)

DESCRIPTION: QNetwork

RESOURCES:
    Based on https://medium.com/@unnatsingh/deep-q-network-with-pytorch-d1ca6f40bfda

"""
import tensorflow as tf
import tensorflow.keras as keras

import config.dqn_settings as dqn_settings

class AtariModel:

    """
    QNetwork - inherits from nn.Module
    """

    def __init__(self):
        self.model = None

    def build_atari_model(self, action_size):

        # With the functional API we need to define the inputs
        frames_input = keras.layers.Input(dqn_settings.ATARI_SHAPE, name='frames')
        actions_input = keras.layers.Input((action_size,), name='mask')

        # Assuming that the input frames are still encoded from 0 to 255, encoding to [0, 1]
        normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)

        # The first hidden layer convolves 16 8x8 filters with stride 4 with the input image
        # and applies a rectifier nonlinearity.
        conv_1 = keras.layers.Conv2D(filters=8, kernel_size=4, strides=4, activation='relu'
                           )(normalized)

        conv_2 = keras.layers.Conv2D(filters=4, kernel_size=1, strides=2, activation='relu')(conv_1)

        conv_flattened = keras.layers.Flatten()(conv_2)

        hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)

        output = keras.layers.Dense(action_size)(hidden)

        filtered_output = keras.layers.Multiply()([output, actions_input])

        self.model = keras.models.Model([frames_input, actions_input], filtered_output)

        optimizer = keras.optimizers.RMSprop(lr=dqn_settings.LEARNING_RATE,
                                             rho=dqn_settings.RHO,
                                             epsilon=0.1)

        self.model.compile(optimizer, loss='mse')



