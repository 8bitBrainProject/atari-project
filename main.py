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

def main():
    """
    main

    Returns
    -------
    None.

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = DqnAgent(state_size=8, action_size=4, seed=0)

if __name__ == "__main__":
    main()
