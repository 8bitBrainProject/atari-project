# -*- coding: utf-8 -*-
"""
COMP 6000 FINAL PROJECT

TEAM: 8-BIT BRAIN

@author: Dennis Brown (dgb0028@auburn.edu)
@author: Shannon McDade (slm0035@auburn.edu)
@author: Jacob Parmer (jdp0061@auburn.edu)

DESCRIPTION: DQN Settings

RESOURCES:
    https://medium.com/@unnatsingh/deep-q-network-with-pytorch-d1ca6f40bfda

"""

# Shape of the atari pixel grid over given number of states
ATARI_SHAPE = (4, 105, 80)

# Number of episodes to run before exiting program
NUMBER_EPISODES = 10000

# Replay Memory Size
MEMORY_SIZE = 25000 

# Minibatch Size
BATCH_SIZE = 32

# Number of times an action will be run before a new action is chosen
ACTIONS_BEFORE_UPDATE = 4

# Number of episodes before the model begins fitting data
ITERATIONS_BEFORE_FIT = 10

# Discount Factor (Gamma)
GAMMA = 0.99

# Soft Update of Target Parameters (Tau)
TAU = 1e-3

# TEMP !!!
RHO = 0.95

# Learning rate
LEARNING_RATE = 1e-3 

# Epsilon values for iteration
INITIAL_EPSILON = 1.00
FINAL_EPSILON = 0.01
