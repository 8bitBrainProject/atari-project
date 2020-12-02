"""
AI FINAL PROJECT

TEAM: 8-BIT BRAIN

@author: Dennis Brown (dgb0028@auburn.edu)
@author: Shannon McDade (slm0035@auburn.edu)
@author: Jacob Parmer (jdp0061@auburn.edu)

DESCRIPTION: DdqnAgent Agent Implementation for Atari 2600 Games
             Currently, will only run Pong
             Heavily inspired by
             https://github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb
"""

import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from collections import deque
from wrappers import make_atari, wrap_deepmind, wrap_pytorch

# Set up CUDA usage
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class ReplayBuffer(object):
    """
    Implement a Replay Buffer for the DDQN algorithm.
    """

    def __init__(self, capacity):
        """
        Set up a Replay Buffer
        """
        self.buffer = deque(maxlen=capacity)


    def push(self, state, action, reward, next_state, done):
        """
        Push S-A-R-S2-Done tuple to the buffer.
        """
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))


    def sample(self, batch_size):
        """
        Pull S-A-R-S2-Done tuple from the buffer.
        """
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done


    def __len__(self):
        """
        Return length of buffer.
        """
        return len(self.buffer)


class CnnDQN(nn.Module):
    """
    Implement a Convolutional Neural Network for the DDQN algorithm.
    """

    def __init__(self, input_shape, num_actions):
        """
        Set up Convolutional Neural Network
        """
        super(CnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )



    def forward(self, x):
        """
        Forward propagation for the fully connected network.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


    def feature_size(self):
        """
        Return feature size.
        """
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)


    def act(self, state, epsilon):
        """
        Choose an action.
        """
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.env.action_space.n)
        return action


class DdqnAgent:
    """
    Agent to execute DDQN for Pong
    """

    def __init__(self):
        self.state_file_name = 'ddqn_pong_model'
        self.log_file_name = 'rewardlog-ddqn.txt'
        self.render = True # Render the game?
        self.forever = True # Run until user kills process?

        # hyperparameters
        self.gamma = 0.99
        self.replay_initial = 10000
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 30000
        self.num_frames = 1000000
        self.batch_size = 32
        self.learning_rate = 0.0001

        # create environment
        self.env_id = "PongNoFrameskip-v4"
        self.env = make_atari(self.env_id)
        self.env = wrap_deepmind(self.env)
        self.env = wrap_pytorch(self.env)

        # create networks
        self.current_model = CnnDQN(self.env.observation_space.shape, self.env.action_space.n)
        self.target_model  = CnnDQN(self.env.observation_space.shape, self.env.action_space.n)
        if USE_CUDA:
            self.current_model = self.current_model.cuda()
            self.target_model  = self.target_model.cuda()

        # setup optimizer
        self.optimizer = optim.Adam(self.current_model.parameters(), lr = self.learning_rate)

        # initialize replay memory
        self.replay_buffer = ReplayBuffer(100000)


    def update_target(self, current_model, target_model):
        """
        Update target network with the current network.
        """
        target_model.load_state_dict(current_model.state_dict())


    def compute_td_loss(self, batch_size):
        """
        Compute and return Temporal Difference loss.
        """
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))

        q_values = self.current_model(state)
        next_q_values = self.current_model(next_state)
        next_q_state_values = self.target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


    def run_ddqn(self):
        """
        Train the Model
        """

        start_time = time.time()

        self.update_target(self.current_model, self.target_model)

        epsilon_by_frame = lambda frame_idx: self.epsilon_final + \
            (self.epsilon_start - self.epsilon_final) * \
                math.exp(-1. * frame_idx / self.epsilon_decay)

        losses = []
        all_rewards = []
        episode_reward = 0
        episode = 0
        reward_log = open(self.log_file_name, 'w')
        running_reward = None
        epsilon = self.epsilon_start

        state = self.env.reset()
        frame_idx = 0
        while (self.forever or (frame_idx < (self.num_frames + 1))):
            frame_idx += 1
            epsilon = epsilon_by_frame(frame_idx)
            action = self.current_model.act(state, epsilon)

            # Render the game if so configured
            if (self.render): self.env.render()

            next_state, reward, done, info = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                # Update the log
                episode += 1
                running_reward = episode_reward if running_reward is None else running_reward * 0.99 + episode_reward * 0.01
                print('steps', frame_idx, 'epsilon', epsilon, 'episode', episode, 'reward', episode_reward, 'running_reward', running_reward)
                reward_log.write(str(episode) + '\t' + str(episode_reward) + '\t' + str(running_reward) + '\t' + str(time.time() - start_time) + '\n')
                reward_log.flush()

                # Reset environment and save rewwards
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0

            # Update losses
            if len(self.replay_buffer) > self.replay_initial:
                loss = self.compute_td_loss(self.batch_size)
                losses.append(loss.data)

            # Update target model
            if frame_idx % 1000 == 0:
                self.update_target(self.current_model, self.target_model)

