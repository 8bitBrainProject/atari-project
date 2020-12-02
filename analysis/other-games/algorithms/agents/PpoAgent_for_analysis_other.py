"""
COMP 6000 FINAL PROJECT

TEAM: 8-BIT BRAIN

@author: Dennis Brown (dgb0028@auburn.edu)
@author: Shannon McDade (slm0035@auburn.edu)
@author: Jacob Parmer (jdp0061@auburn.edu)

DESCRIPTION: PpoAgent Agent Implementation for Atari 2600 Games
             Currently, will only run Pong
RESOURCES:
"""

import random
import gym
import numpy as np
import torch
import time
from torch.nn import functional as F
from torch import nn

from agents.wrappers import make_atari, wrap_deepmind  # DGB

import config.ppo_settings as ppo_settings

class PpoAgent(nn.Module):
    """
    CLASS: PpoAgent
    DESCRIPTION:
    """

    def __init__(self):
        """
        __init__

        Returns
        -------
        None.

        """
        # Inheriting from the nn.Module, which is the base class for all neural
        # network modules in PyTorch
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        super(PpoAgent, self).__init__()

        # Importing Settings for the PpoAgent
        self.gamma = ppo_settings.GAMMA
        self.episode_clip = ppo_settings.EPISODE_CLIP

        # Defining a sequential container
        # https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        self.layers = nn.Sequential(

            # Applies linear transformation to incoming data
            # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
            nn.Linear(7056, 512), # DGB 6000 to 7056

            # Applies rectified linear unit function element-wise
            # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            nn.ReLU(),

            # Applies linear transformation to incoming data (see above link)
            nn.Linear(512, 2)
            )

    def state_to_tensor(self, image):
        """
        state_to_tensor

        Parameters
        ----------
        I : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if image is None:
            # Returning a tensor filled with the scalar value 0, with the shape
            # defined by the variable argument size
            # https://pytorch.org/docs/stable/generated/torch.zeros.html
            return torch.zeros(1, 7056) # DGB 6000 to 7056

        # DGB -- commented out this block
        # # Removing 35 pixels from the start & 25 pixels from the end
        # # Getting rid of parts of image that are not needed
        # image = image[35:185]

        # # Downsampling by a factor of two
        # image = image[::2, ::2, 0]

        # # Erase background (background type 1)
        # image[image == 144] = 0

        # # Erase background (background type 2)
        # image[image == 109] = 0

        # # Grayscale
        # image[image != 0] = 1


        # Returning a tensor from a numpy.ndarray
        # https://pytorch.org/docs/stable/generated/torch.from_numpy.html.rav
        # torch.unsqueeze():
        # https://pytorch.org/docs/stable/generated/torch.from_numpy.html
        # numpy.ravel() returns a contiguous flattened array
        # https://numpy.org/doc/stable/reference/generated/numpy.ravel.html
        return torch.from_numpy(image.astype(np.float32).ravel()).unsqueeze(0)

    def preprocess(self, image, prev_image):
        """
        preprocess

        Parameters
        ----------
        image : TYPE
            DESCRIPTION.
        prev_image : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self.state_to_tensor(image) - self.state_to_tensor(prev_image)

    def convert_action(self, action):
        """
        convert_action

        Parameters
        ----------
        action : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return action + 2

    def forward(self, state, action=None, action_probability=None, advantage=None,
                deterministic=None):
        """
        forward

        Parameters
        ----------
        state : TYPE
            DESCRIPTION.
        action : TYPE, optimizerional
            DESCRIPTION. The default is None.
        action_prob : TYPE, optimizerional
            DESCRIPTION. The default is None.
        advantage : TYPE, optimizerional
            DESCRIPTION. The default is None.
        deterministic : TYPE, optimizerional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        if action is None:

            # Disabling gradient calculation
            # https://pytorch.org/docs/stable/generated/torch.no_grad.html
            with torch.no_grad():

                # Logit def:
                # https://en.wikipedia.org/wiki/Logit
                logits = self.layers(state)

                if deterministic:
                    # Returning the indices of the maximum value of all elements
                    # in the input tensor for action
                    # https://pytorch.org/docs/stable/generated/torch.argmax.html
                    action = int(torch.argmax(logits[0]).detach().cpu().numpy())

                    # Setting the action probability to 1.0
                    action_probability = 1.0

                else:
                    # Creating a categorical distribution parameterized by logits
                    # https://pytorch.org/docs/stable/distributions.html
                    c = torch.distributions.Categorical(logits=logits)

                    # Sample an action from the output of the network
                    action = int(c.sample().cpu().numpy()[0])

                    # Setting the action probability
                    action_probability = float(c.probs[0, action].detach().cpu().numpy())

                return action, action_probability

        # Creating the vs numpy array
        vs = np.array([[1.0, 0.0],
                       [0.0, 1.0]])

        # Specifying torch.FloatTensor type
        ts = torch.FloatTensor(vs[action.cpu().numpy()])

        logits = self.layers(state)

        # Returning the sum of all elements in the input tensor
        # https://pytorch.org/docs/stable/generated/torch.sum.html
        # F is a torch.nn.functional:
        # https://pytorch.org/docs/stable/nn.functional.html
        # Softmax:
        # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html#torch.nn.Softmax
        r = torch.sum(F.softmax(logits, dim=1) * ts, dim=1) / action_probability

        # Calculating the loss
        loss1 = r * advantage
        # Clamps all elements in input into the range [min, max] and returns a
        # resulting tensor
        # https://pytorch.org/docs/stable/generated/torch.clamp.html
        loss2 = torch.clamp(r, 1-self.episode_clip, 1+self.episode_clip) * advantage
        # Returning the minimum value of all elements in the input tensor
        loss = -torch.min(loss1, loss2)
        # Returning the mean of all elements in the input tensor
        loss = torch.mean(loss)

        # Returning the calculated loss
        return loss

# =============================================================================
# END PpoAgent CLASS -- START run_ppo
# =============================================================================

def run_ppo(policy, game_env, log_token, run_no):
    """
    run_ppo

    Returns
    -------
    None.

    """

    start_time = time.time()

    env = make_atari(game_env)
    env = wrap_deepmind(env)
    env.reset()

    # Constructing an optimizerimizer
    # https://pytorch.org/docs/stable/optim.html
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4) # DGB 1e-3 to 1e-4

    # Initializing running_average to None
    running_average = None

    reward_log = open('ppo ' + log_token + ' 1e-4 no' + str(run_no) + '.txt', 'w')

    # for each iteration
    for it in range(25):
        d_obs_history, action_history, action_prob_history, reward_history = [], [], [], []

        # Each iteration will have 10 episodes
        for episode in range(10):
            obs, prev_obs = env.reset(), None

            for t in range(190000):
                env.render()

                # Preprocessing the image
                d_obs = policy.preprocess(obs, prev_obs)

                # Disabling gradient calculation
                # https://pytorch.org/docs/stable/generated/torch.no_grad.html
                with torch.no_grad():
                    action, action_prob = policy(d_obs)

                prev_obs = obs
                obs, reward, done, _ = env.step(policy.convert_action(action))

                d_obs_history.append(d_obs)
                action_history.append(action)
                action_prob_history.append(action_prob)
                reward_history.append(reward)

                if done:
                    reward_sum = sum(reward_history[-t:])
                    running_average = (0.99
                                       * running_average
                                       + 0.01
                                       * reward_sum if running_average else reward_sum)

                    print(('Iteration %d, Episode %d (%d timesteps) - '
                           + 'last_action: %d, last_action_prob: %.2f, '
                           + 'reward_sum: %.2f, running_avg: %.2f')
                          % (it, episode, t, action, action_prob, reward_sum,
                             running_average))

                    reward_log.write(str(it) + '\t'
                                      + str(episode) + '\t'
                                      + str(t) + '\t'
                                      + str(action) + '\t'
                                      + str(action_prob) + '\t'
                                      + str(reward_sum) + '\t'
                                      + str(running_average) + '\t'
                                      + str(time.time() - start_time) + '\n')

                    # reward_log.write(str((it * 10) + episode) + '\t'
                    #                  + str(reward_sum) + '\t'
                    #                  + str(running_average) + '\t'

                    reward_log.flush()

                    break

        # compute advantage
        R = 0
        discounted_rewards = []

        for r in reward_history[::-1]:
            if r != 0:
                R = 0 # scored/lost a point in pong, so reset reward sum
            R = r + policy.gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards
                              - discounted_rewards.mean()) / discounted_rewards.std()

        # update policy
        for _ in range(5):
            n_batch = 500  # DGB 24576 reduced to 5000 since action history is smaller now (reduced 34075 -> 7502)

            # Random number
            idxs = random.sample(range(len(action_history)), n_batch)

            # Concatenate the given sequence of tensors in the given dimension
            # https://pytorch.org/docs/stable/generated/torch.cat.html
            d_obs_batch = torch.cat([d_obs_history[idx] for idx in idxs], 0)

            # Specifying torch.LongTensor type, a multi-dimensional matrix containing
            # elements of a single data type. Each type has CPU and GPU variants
            # https://pytorch.org/docs/stable/tensors.html
            action_batch = torch.LongTensor([action_history[idx] for idx in idxs])

            # Specifying torch.FloatTensor type
            action_prob_batch = torch.FloatTensor([action_prob_history[idx] for idx in idxs])
            advantage_batch = torch.FloatTensor([discounted_rewards[idx] for idx in idxs])

            optimizer.zero_grad()
            loss = policy(d_obs_batch, action_batch, action_prob_batch, advantage_batch)
            loss.backward()
            optimizer.step()

        # if it % 5 == 0:
        #     torch.save(policy.state_dict(), 'params.ckpt')

    env.close()
