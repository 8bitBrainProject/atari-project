"""
COMP 5600/6600/6606 FINAL PROJECT

TEAM: 8-BIT BRAIN

@author: Dennis Brown (dgb0028@auburn.edu)
@author: Shannon McDade (slm0035@auburn.edu)
@author: Jacob Parmer (jdp0061@auburn.edu)

DESCRIPTION: Random Agent Implementation for Atari 2600 Games
             Currently, will only run Pong
"""

import numpy as np
import time
from agents.wrappers import make_atari, wrap_deepmind


class RandomAgent:
    """
    Plays Pong with random actions.
    """
    def __init__(self):
        self.render = True
        pass

    def run_random(self):
        """
        Randomly play the game. Log results.
        """
        start_time = time.time()

        # Create environment
        env = make_atari("PongNoFrameskip-v4")
        env = wrap_deepmind(env)
        observation = env.reset()

        # bookkeeping values
        running_reward = None
        reward_sum = 0
        episode_number = 0

        # Open load
        reward_log = open('rewardlog-random.txt', 'w')

        while (True):  # Run until user kills process
            if self.render: env.render()

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

