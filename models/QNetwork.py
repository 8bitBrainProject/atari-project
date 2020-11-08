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
        #self.fc1 = nn.Linear(state_size, fc1)
        #self.fc2 = nn.Linear(fc1, fc2)
        #self.fc3 = nn.Linear(fc2, action_size)

        self.conv1 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=16, stride=8)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=8, stride=4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(128, 1)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(1, action_size)


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
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out
        #return self.fc3(x)
