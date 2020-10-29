# -*- coding: utf-8 -*-
"""
COMP 6000 FINAL PROJECT

TEAM: 8-BIT BRAIN

@author: Dennis Brown (dgb0028@auburn.edu)
@author: Shannon McDade (slm0035@auburn.edu)
@author: Jacob Parmer (jdp0061@auburn.edu)

DESCRIPTION: QNetwork

RESOURCES:
    Based on https://medium.com/@unnatsingh/deep-q-network-with-pytorch-d1ca6f40bfda

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# !!! Update as you see fit, this is taken from the medium article above
class QNetwork(nn.Module):
    """
    QNetwork - inherits from nn.Module
    """

    def __init__(self, state_size, action_size, seed, fc1=64, fc2=64):
        """
        Initialization

        Parameters
        ----------
        state_size : TYPE
            DESCRIPTION.
        action_size : TYPE
            DESCRIPTION.
        seed : TYPE
            DESCRIPTION.
        fc1 : TYPE, optional
            DESCRIPTION. The default is 64.
        fc2 : TYPE, optional
            DESCRIPTION. The default is 64.

        Returns
        -------
        None.

        """

        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, action_size)

    def forward(self, x):
        """


        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)
