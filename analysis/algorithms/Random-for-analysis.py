"""
COMP 5600/6600/6606 FINAL PROJECT

TEAM: 8-BIT BRAIN

@author: Dennis Brown (dgb0028@auburn.edu)
@author: Shannon McDade (slm0035@auburn.edu)
@author: Jacob Parmer (jdp0061@auburn.edu)

Play games with random actions and log results.
"""

import numpy as np
import time
import sys
import gym
from wrappers import make_atari, wrap_deepmind

render = False

if __name__ == '__main__':
    start_time = time.time()

    env = make_atari("PongNoFrameskip-v4")
    #env = make_atari("SpaceInvadersNoFrameskip-v4")
    #env = make_atari("MsPacmanNoFrameskip-v4")
    env = wrap_deepmind(env)
    observation = env.reset()
    running_reward = None
    reward_sum = 0
    episode_number = 0

    reward_log = open('random pv4w 500 no' + str(sys.argv[1]) + '.txt', 'w')

    while (episode_number < 500):
        if render: env.render()

        # Flip a coin to choose UP (action = 2) or DOWN (action = 3)
        action = 2
        if (np.random.uniform() < 0.5): action = 3

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        if done: # an episode finished
            episode_number += 1

            # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('episode:', episode_number, 'reward:', reward_sum, 'running reward:', running_reward)
            reward_log.write(str(episode_number) + '\t' + str(reward_sum) + '\t' + str(running_reward) + '\t' + str(time.time() - start_time) + '\n')
            reward_log.flush()
            reward_sum = 0
            observation = env.reset() # reset env

