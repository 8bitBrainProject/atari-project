# https://github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb

# "ANALYSIS" version of code is hard-coded to run 500 episodes, save
# timing information, never save network state, and save log to hardcoded
# filename + run number supplied as single command line argument.

import sys
import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

import gym
from collections import deque
from wrappers import make_atari, wrap_deepmind, wrap_pytorch


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)


    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))

    q_values      = current_model(state)
    next_q_values = current_model(next_state)
    next_q_state_values = target_model(next_state)

    q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def train(forever = False, render = False):

    update_target(current_model, target_model)

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

    losses = []
    all_rewards = []
    episode_reward = 0

    #---------------------
    episode = 0
    reward_log = open(log_file_name, 'w')
    running_reward = None
    epsilon = epsilon_start
    #---------------------

    state = env.reset()
    frame_idx = 0
    while (episode < 250): #(forever or (frame_idx < (num_frames + 1))):
        frame_idx += 1
        epsilon = epsilon_by_frame(frame_idx)
        action = current_model.act(state, epsilon)

        #---------------------
        if (render): env.render()
        #---------------------

        next_state, reward, done, info = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            #---------------------
            episode += 1
            running_reward = episode_reward if running_reward is None else running_reward * 0.99 + episode_reward * 0.01
            print('steps', frame_idx, 'epsilon', epsilon, 'episode', episode, 'reward', episode_reward, 'running_reward', running_reward)
            reward_log.write(str(episode) + '\t' + str(episode_reward) + '\t' + str(running_reward) + '\t' + str(time.time() - start_time) + '\n')
            reward_log.flush()
            #---------------------
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > replay_initial:
            loss = compute_td_loss(batch_size)
            #---------------------
            # losses.append(loss.data[0])
            losses.append(loss.data)   # See https://github.com/NVIDIA/flownet2-pytorch/issues/113
            #---------------------

        if frame_idx % 1000 == 0:
            update_target(current_model, target_model)

        # if ((frame_idx % 25000) == 0):
        #     torch.save(current_model, state_file_name)


def test(forever = False, render = False):
    frame_idx = 0
    episode = 0
    running_reward = None
    episode_reward = 0

    state = env.reset()
    while (forever or (frame_idx < (num_frames + 1))):
        if render: env.render()

        frame_idx += 1

        action = current_model.act(state, 0.01)  # set epsilon to 0.0 to eventually get stuck in stalemate and break game
        next_state, reward, done, _ = env.step(action)
        state = next_state
        episode_reward += reward

        if done:
            episode += 1
            running_reward = episode_reward if running_reward is None else running_reward * 0.99 + episode_reward * 0.01
            print('steps', frame_idx, 'episode', episode, 'reward', episode_reward, 'running_reward', running_reward)
            state = env.reset()
            episode_reward = 0


if __name__ == '__main__':
    start_time = time.time()

    # mode = 'test'
    # if (len(sys.argv) >= 2):
    #     mode = sys.argv[1]
    #     print('mode:', mode)
    # else:
    #     print('Did not specify mode; using', mode)

    # state_file_name = 'dqn_pong_model'
    # if (len(sys.argv) >= 3):
    #     state_file_name = sys.argv[2]
    #     print('state_file_name:', state_file_name)
    # else:
    #     print('Did not specify state_file_name; using', state_file_name)

    # log_file_name = 'rewardlog.txt'
    # if (mode == 'train'):
    #     if (len(sys.argv) >= 4):
    #         log_file_name = sys.argv[3]
    #         print('log_file_name:', log_file_name)
    #     else:
    #         print('Did not specify log_file_name; using', log_file_name)

    log_file_name = 'ddqn ' + sys.argv[2] + '1e-4 250 no' + sys.argv[3] + '.txt'

    # hyperparameters
    gamma = 0.99
    replay_initial = 10000
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 30000
    num_frames = 1000000
    batch_size = 32
    learning_rate = 0.0001

    # create environment
    # env_id = "PongNoFrameskip-v4"
    # env_id = 'SpaceInvadersNoFrameskip-v4'
    # env_id = 'MsPacmanNoFrameskip-v4'
    # env_id = 'VideoPinballNoFrameskip-v4'
    # env_id = 'MontezumaRevengeNoFrameskip-v4'
    # env_id = 'QbertNoFrameskip-v4'
    env_id = sys.argv[1]
    env    = make_atari(env_id)
    # env = gym.wrappers.Monitor(env, 'stats', video_callable=lambda episode_id: False, force=True, resume=False)
    env    = wrap_deepmind(env)
    env    = wrap_pytorch(env)

    # create networks
    current_model = CnnDQN(env.observation_space.shape, env.action_space.n)
    target_model  = CnnDQN(env.observation_space.shape, env.action_space.n)
    if USE_CUDA:
        current_model = current_model.cuda()
        target_model  = target_model.cuda()

    # setup optimizer
    optimizer = optim.Adam(current_model.parameters(), lr = learning_rate)

    # initialize replay memory
    replay_buffer = ReplayBuffer(100000)

    # train model
    # if (mode == 'train'):
    train(forever = True, render = True)

    # # test model
    # if (mode == 'test'):
    #     current_model = torch.load(state_file_name)
    #     test(forever = True, render = True)

