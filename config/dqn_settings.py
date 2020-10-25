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

# Replay Buffer Size
BUFFER_SIZE = int(1e5)

# Minibatch Size
BATCH_SIZE = 64

# Discount Factor (Gamma)
GAMMA = 0.99

# Soft Update of Target Parameters (Tau)
TAU = 1e-3

# Learning rate
LEARNING_RATE = 5e-4

# Frequency to update the network (seconds)
UPDATE_FREQUENCY = 4
