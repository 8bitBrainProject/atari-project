# -*- coding: utf-8 -*-
"""
COMP 5600/6600/6606 FINAL PROJECT

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
from agents.PpoAgent import PpoAgent, run_ppo
from agents.DdqnAgent import DdqnAgent
from agents.PgAgent import PgAgent
from agents.RandomAgent import RandomAgent


def main(ppo=False, dqn=False, ddqn=False, pg=False, random=False):
    """
    main

    Parameters
    ----------
    ppo : BOOLEAN, optional
        SET TO TRUE TO RUN PPO AGENT. The default is False.
    dqn : BOOLEAN, optional
        SET TO TRUE TO RUN DQN AGENT. The default is False.
    ddqn : BOOLEAN, optional
        SET TO TRUE TO RUN DDQN AGENT. The default is False.
    pg : BOOLEAN, optional
        SET TO TRUE TO RUN PG AGENT. The default is False.
    random : BOOLEAN, optional
        SET TO TRUE TO RUN RANDOM AGENT. The default is False.

    Returns
    -------
    None.

    """

    # Running DQN Agent
    if dqn:
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

    # Running PPO Agent
    elif ppo:
        policy = PpoAgent()
        run_ppo(policy)

    # Running DDQN Agent
    elif ddqn:
        ddqn_agent = DdqnAgent()
        ddqn_agent.run_ddqn()

    # Running PG Agent
    elif pg:
        pg_agent = PgAgent()
        pg_agent.run_pg()

    # Running Random Agent
    elif random:
        random_agent = RandomAgent()
        random_agent.run_random()

    # If neither agent is selected a message will print asking to select one
    else:
        print('No agent selected to run! Please select an agent: dqn, ppo, ddqn, pg, random')

if __name__ == "__main__":
    # Set one of these to True
    main(ppo=False, dqn=False, ddqn=False, pg=True, random=False)
