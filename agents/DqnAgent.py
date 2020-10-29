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
    def __init__(self, state_size, action_size, device, seed, number_episodes=200, max_t=1000):
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
        self.number_episodes = number_episodes
        self.max_t = max_t
        self.seed = random.seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.memory = None # will initialize in dqn_search

        # Local QNetwork
        self.qnetwork_local = QNetwork(self.state_size,
                                       self.action_size, seed).to(device)

        # Target QNetwork
        self.qnetwork_target = QNetwork(self.state_size,
                                        self.action_size, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),
                                    lr=dqn_settings.LEARNING_RATE)

    def choose_action(self, state, epsilon):
        """


        Parameters
        ----------
        state : TYPE
            DESCRIPTION.
        epsilon : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

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
                                  cv2.COLOR_BGR2GRAY).astype(np.uint8)
        image_data[image_data > 0] = 255


        image_tensor = image.transpose(2, 0, 1)
        image_tensor = torch.from_numpy(image_tensor)
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        return image_tensor    

    def replay_memory(self, action_size, buffer_size, batch_size, seed):
        """
        replay_memory

        Parameters
        ----------
        action_size : TYPE
            DESCRIPTION.
        buffer_size : TYPE
            DESCRIPTION.
        batch_size : TYPE
            DESCRIPTION.
        seed : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # !!! See ReplayBuffer example from medium as example for creation
        # !!! We can break this off into a class if you want -- just think this
        #     can simply be store what we want in a list of lists possibly??
        #     similar to: https://github.com/nevenp/dqn_flappy_bird/blob/master/dqn.py
        # !!! Can change parameters we are saving

    # !!! I made game_score optional for now per discussion ... can delete if want to

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
        # Initialize replay memory

        # Initialize action-value function Q with random weight Theta

        # Initialize target action-value function Q(^) with weights
        # Theta(-) = Theta

        # for episode = 1 to self.number_episodes do

            # Initialize sequence s_1 = {x_1} and preprocessed sequence
            # Phi_1 = Phi(s_1)

            # for t = 1 to self.max_t do

                # Following epsilon-greedy policy, select a_t = { a random
                # action ... with probability epsilon, and arg maxa Q(Phi(s_t), a;
                # Theta) ... otherwise

                # Execute action a_i in emulator and observe reward r_t and image
                # x_t+1

                # Set s_t+1 = s_t, a_t, x_t+1, and preprocess Phi_t+1 = Phi(s_t+1)

                # Store transition (Phi_t, a_t, r_t, Phi_t+1) in D

                # Experience Replay
                # Sample random minibatch of transitions (Phi_j, a_j, r_j, Phi_j+1)
                # from D

                # Set y_j = { r_j ... if episode terminates at step j+1 and
                # r_j+gamma max_a'Q(^)(Phi_j+1, a';Theta(-)) ... otherwise

                # Perform a gradient descent step on (y_j - Q(Phi_j, a_j; Theta))^2
                # w.r.t. the network parameter Theta

                # Periodic Update of Target Network
                # Every C steps reset Q(^) = Q, i.e., set Theta(-) = Theta

            # end
        # end
