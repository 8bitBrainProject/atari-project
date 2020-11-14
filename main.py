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
import tensorflow as tf
import gym

import config.dqn_settings as dqn_settings
from agents.DqnAgent import DqnAgent
from agents.RingBuffer import RingBuffer
from models.QNetwork import AtariModel

def main():
    """
    main

    Returns
    -------
    None.

    """

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
        
if __name__ == "__main__":
    main()
