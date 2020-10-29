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
import gym
from agents.DqnAgent import DqnAgent

def main():
    """
    main

    Returns
    -------
    None.

    """
    env = gym.make('BreakoutDeterministic-v4')
    frame = env.reset()


    state_size = 8 
    print(state_size)
    action_size = env.action_space.n
    print(action_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = DqnAgent(state_size=state_size, action_size=action_size, device=device, seed=0)


    env.render()

    finished = False
    while not finished:
        
        action = env.action_space.sample()
        frame, reward, finished, _ = env.step(action)
        env.render()


if __name__ == "__main__":
    main()
