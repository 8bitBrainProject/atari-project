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

NUMBER_EPISODES = 200

def main():
    """
    main

    Returns
    -------
    None.

    """

    env = gym.make('SpaceInvadersNoFrameskip-v4')

    state_size = 4 
    action_size = env.action_space.n

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = DqnAgent(state_size=state_size, action_size=action_size, device=device, seed=0)

    for i in range(NUMBER_EPISODES):

        frame = env.reset()
        image_data = agent.image_preprocessing(frame)

        state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

        env.render()

        finished = False
        
        while not finished:
        
            action = agent.choose_action(state)
            final_epsilon_reached = agent.iterate_epsilon()

            frame, reward, finished, _ = env.step(action)

            frame = agent.image_preprocessing(frame)
            next_state = torch.cat((state.squeeze(0)[3:, :, :], frame)).unsqueeze(0)

            state = next_state

            env.render()

if __name__ == "__main__":
    main()
