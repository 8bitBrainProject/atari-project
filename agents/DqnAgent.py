# -*- coding: utf-8 -*-
"""
COMP 5600/6600/6606 FINAL PROJECT

TEAM: 8-BIT BRAIN

@author: Dennis Brown (dgb0028@auburn.edu)
@author: Shannon McDade (slm0035@auburn.edu)
@author: Jacob Parmer (jdp0061@auburn.edu)

****This is just a skeleton of a dqn agent --- add/delete/modify as needed!****

DESCRIPTION: DQN Agent Implementation for Atari 2600 Games

RESOURCES:
    >>> Mnih et. al.
    >>> https://medium.com/@unnatsingh/deep-q-network-with-pytorch-d1ca6f40bfda
    >>> https://github.com/nevenp/dqn_flappy_bird/blob/master/dqn.py

"""
import random
import numpy as np
import tensorflow as tf

import config.dqn_settings as dqn_settings
from models.QNetwork import AtariModel

class DqnAgent():
    """
    DqnAgent - A Deep-Q Network Agent implementation
    """

    # TIP: to generate docstring automatically highlight function name
    #      (e.g., dqn_search), right-click, and select 'Generate docstring'
    def __init__(self, action_size):
        """
        Initialization

        Parameters
        ----------
        state_size : int
            DESCRIPTION.
        action_size : int
            Number of possible actions at a given state for the agent to take.
        device : TYPE
            DESCRIPTION.
        seed : TYPE
            DESCRIPTION.
        number_episodes : TYPE, optional
            DESCRIPTION. The default is 200.
        max_t : TYPE, optional
            DESCRIPTION. The default is 1000.

        Returns
        -------
        None.

        """
        self.epsilon = dqn_settings.INITIAL_EPSILON
        self.action_size = action_size

    def choose_action(self, state, model):
        """


        Parameters
        ----------
        state :
            DESCRIPTION.

        Returns
        -------
        One hot action array.

        """

        if (random.random() < self.epsilon):

            action = random.randrange(0, self.action_size)

        else:

            np_states = np.array(state)[None, :, :, :]
            np_actions = np.ones((1, self.action_size))

            next_Q_values = model.predict([np_states, np_actions])
            action = np.argmax(next_Q_values, axis=1)[0]

        return action

    def image_preprocessing(self, image):
        """
        image_preprocessing

        Parameters
        ----------
        image : TYPE
            DESCRIPTION.

        Returns
        -------
        image_tensor : numpy array
            A much smaller and more computationally efficient image, turned into a tensor for
            use in a PyTorch neural network.

        """
        # See image_to_tensor & resize_and_bgr2gray from
        # https://github.com/nevenp/dqn_flappy_bird/blob/master/dqn.py

        image = np.mean(image, axis=2).astype(np.uint8)
        image = image[::2, ::2]
        return image

    def transform_reward(self, reward):
        return np.sign(reward)

    def fit_batch(self, model, batch, action_size):

        start_states = np.array(tuple(d[0] for d in batch))
        actions = tuple(d[1] for d in batch)
        next_states = np.array(tuple(d[2] for d in batch))
        rewards = np.array(tuple(d[3] for d in batch))
        finished = tuple(d[4] for d in batch)

        onehot_actions = np.zeros((len(batch), action_size))

        for i, action_index in enumerate(actions):
            onehot_actions[i][action_index] = 1

        next_Q_values = model.predict([next_states, np.ones(onehot_actions.shape)])

        next_Q_values[finished] = 0

        Q_values = rewards + dqn_settings.GAMMA * np.max(next_Q_values, axis=1)

        Q_values_temp = np.array(Q_values).reshape((1, len(next_Q_values)))
        Q_transposed = np.transpose(Q_values_temp)

        model.fit([start_states, onehot_actions], onehot_actions * Q_transposed,
                  epochs=1, batch_size=len(start_states), verbose=0)
        return
