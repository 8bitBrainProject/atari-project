# -*- coding: utf-8 -*-
"""
COMP 6000 FINAL PROJECT

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
import cv2

import torch
import torch.nn.functional as F
import torch.optim as optim

import config.dqn_settings as dqn_settings
from models.QNetwork import QNetwork

class DqnAgent():
    """
    DqnAgent - A Deep-Q Network Agent implementation
    """

    # TIP: to generate docstring automatically highlight function name
    #      (e.g., dqn_search), right-click, and select 'Generate docstring'
    def __init__(self, state_size, action_size, device, seed, max_t=1000):
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
        self.max_t = max_t
        self.seed = random.seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = dqn_settings.INITIAL_EPSILON
        self.final_epsilon = dqn_settings.FINAL_EPSILON
        self.iteration_speed = dqn_settings.GAMMA
        self.memory = None # will initialize in dqn_search
        self.replay_memory = []

        # Local QNetwork
        self.qnetwork_local = QNetwork(self.state_size,
                                       self.action_size, seed).to(device)

        # Target QNetwork
        self.qnetwork_target = QNetwork(self.state_size,
                                        self.action_size, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),
                                    lr=dqn_settings.LEARNING_RATE)

    def choose_action(self, state):
        """


        Parameters
        ----------
        state :
            DESCRIPTION.

        Returns
        -------
        One hot action array.

        """

        #action = torch.zeros([self.action_size], dtype=torch.uint8)

        output = self.qnetwork_local(state)[0]
        
        if (random.random() < self.epsilon):
            action = random.randrange(0, (self.action_size - 1))
            print("Random action taken!\n")
        else:
            action = torch.argmax(output)
            print(action)

        return action

    def iterate_epsilon(self):
        """

        Parameters
        ----------

        Returns
        -------
        None.


        """
        self.epsilon = self.epsilon * self.iteration_speed
        return self.epsilon <= self.final_epsilon
            

    def predict_next_state(self, state):
        """


        Parameters
        ----------
        state : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

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

        image = image[16:288, 0:404] # Value of 16 cuts off the scoreboard
        image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), 
                                  cv2.COLOR_BGR2GRAY)
        image_data[image_data > 0] = 255


        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor.astype(np.float32)
        image_tensor = torch.from_numpy(image_tensor)
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        return image_tensor    

    def dqn_search(self, pixels, game_score=None):
        """
        dqn_search is adapted from Deep-Q Network (DQN), Mnih et. al. (2015)

        Parameters
        ----------
        pixels : TYPE
            DESCRIPTION.
        game_score : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        output = self.q
