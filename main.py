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
# import tensorflow as tf
import gym

import config.dqn_settings as dqn_settings
from agents.DqnAgent import DqnAgent
from agents.RingBuffer import RingBuffer
from models.QNetwork import AtariModel
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
        env = gym.make('PongDeterministic-v4')
        action_size = env.action_space.n
        agent = DqnAgent(action_size)
        nn = AtariModel()
        nn.build_atari_model(action_size)
        model = nn.model
        memory = RingBuffer(dqn_settings.MEMORY_SIZE)

        for i in range(dqn_settings.NUMBER_EPISODES):
            frame = env.reset()
            frame = agent.image_preprocessing(frame)
            state = (frame, frame, frame, frame)
            env.render()
            finished = False
            summed_reward = 0

            while not finished:
                action = agent.choose_action(state, model)
                next_frame, reward, finished, _ = env.step(action)
                next_frame = agent.image_preprocessing(next_frame)
                reward = agent.transform_reward(reward)
                next_state = (next_frame, state[0], state[1], state[2])
                summed_reward += reward
                memory.append((state, action, next_state, reward, finished))
                state = next_state
                env.render()

                if (i > dqn_settings.ITERATIONS_BEFORE_FIT):
                    minibatch = memory.sample_random_batch(dqn_settings.BATCH_SIZE)
                    agent.fit_batch(model, minibatch, action_size)

            if (agent.epsilon > dqn_settings.FINAL_EPSILON):
                agent.epsilon = agent.epsilon * dqn_settings.GAMMA
            print("Iteration:", i, " Reward:", summed_reward, "Epsilon:", agent.epsilon)
            f = open("rewards.txt", "a")
            f.write(str(summed_reward) + "\n")
            f.close()
            if (i % 100 == 0):
                model.save('models/saved_model')


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


    # If no agent is selected a message will print asking to select one
    else:
        print('No agent selected to run! Please select an agent: dqn, ppo, ddqn, pg, random')


if __name__ == "__main__":
    # Set one of these to True
    main(ppo=True, dqn=False, ddqn=False, pg=False, random=False)
