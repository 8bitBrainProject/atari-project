# -*- coding: utf-8 -*-
"""
COMP 6000 FINAL PROJECT

TEAM: 8-BIT BRAIN

@author: Dennis Brown (dgb0028@auburn.edu)
@author: Shannon McDade (slm0035@auburn.edu)
@author: Jacob Parmer (jdp0061@auburn.edu)

DESCRIPTION: main - run everything from here

RESOURCES:

"""
import torch
from agents.DqnAgent import DqnAgent

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DqnAgent(device)
