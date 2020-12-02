"""
COMP 5600/6600/6606 FINAL PROJECT

TEAM: 8-BIT BRAIN

@author: Dennis Brown (dgb0028@auburn.edu)
@author: Shannon McDade (slm0035@auburn.edu)
@author: Jacob Parmer (jdp0061@auburn.edu)

DESCRIPTION: PpoAgent Agent Implementation for Atari 2600 Games
             A rewardlog is output with our results for data processing/plotting
             later. I included extensive comments with links to documentation
             to assist me in the learning process of using PyTorch & also to help
             any others who are new to using PyTorch.
RESOURCES:
    > This agent was inspired by and modeled after the PPO agent of Sagar Gubbi
        Source: https://www.sagargv.com/blog/pong-ppo/
"""
import random
import gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import config.ppo_settings as ppo_settings

class PpoAgent(nn.Module):
    """
    CLASS:          PpoAgent
    DESCRIPTION:    A Proximal Policy Optimization (PPO) Agent Implementation
    NOTES:          Inherits from the PyTorch nn.Module
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

        # Importing Settings for the PpoAgent from ppo_settings
        self.gamma = ppo_settings.GAMMA
        self.episode_clip = ppo_settings.EPISODE_CLIP
        self.learning_rate = ppo_settings.LEARNING_RATE
        self.step_size = ppo_settings.STEP_SIZE

        # Defining a sequential container
        # https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        self.layers = nn.Sequential(

            # Applies linear transformation to incoming data
            # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
            nn.Linear(ppo_settings.IN_FEATURES1, ppo_settings.OUT_FEATURES1),

            # Applies rectified linear unit function element-wise
            # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            nn.ReLU(),

            # Applies linear transformation to incoming data (see above link)
            nn.Linear(ppo_settings.IN_FEATURES2, ppo_settings.OUT_FEATURES2)
            )

    def process_image(self, image):
        """
        process_image

        Parameters
        ----------
        image : TENSOR
            IMAGE TENSOR.

        Returns
        -------
        TENSOR
            RETURNS A TENSOR FROM A NUMPY.NDARRAY.

        """


        if image is None:
            # Returning a tensor filled with the scalar value 0, with the shape
            # defined by the variable argument size
            # https://pytorch.org/docs/stable/generated/torch.zeros.html
            return torch.zeros(1, 6000)

        # Cropping the image to remove unneccessary parts
        image = image[ppo_settings.CROP_BEGINNING:ppo_settings.CROP_END]

        # Downsampling
        image = image[::ppo_settings.DOWNSAMPLE_FACTOR,
                      ::ppo_settings.DOWNSAMPLE_FACTOR, 0]

        # Erasing background
        image[image == ppo_settings.BACKGROUND_ERASE1] = 0
        image[image == ppo_settings.BACKGROUND_ERASE2] = 0

        # Grayscale
        image[image != 0] = 1

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
        image : TENSOR
            IMAGE.
        prev_image : TENSOR
            IMAGE.

        Returns
        -------
        delta_states : TENSOR
            DELTA_STATES.

        """

        # Processing the current image/state
        processed_image = self.process_image(image)

        # Processing the previous image/state
        previous_image_processed = self.process_image(prev_image)

        # Getting the delta between states
        delta_states = processed_image - previous_image_processed

        return delta_states

    def get_next_step(self, step):
        """
        get_next_step

        Parameters
        ----------
        step : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        next_step = step + self.step_size

        return next_step

    def forward(self, state, action=None, action_probability=None, advantage=None,
                deterministic=None):
        """
        forward -- this method is required for PyTorch implementation

        Parameters
        ----------
        state : TYPE
            DESCRIPTION.
        action : TYPE, optional
            DESCRIPTION. The default is None.
        action_probability : TYPE, optional
            DESCRIPTION. The default is None.
        advantage : TYPE, optional
            DESCRIPTION. The default is None.
        deterministic : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

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
                    cat_distro = torch.distributions.Categorical(logits=logits)

                    # Sample an action from the output of the network
                    action = int(cat_distro.sample().cpu().numpy()[0])

                    # Setting the action probability
                    action_probability = float(cat_distro.probs[0, action].\
                                               detach().cpu().numpy())

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
        calculated_loss = -torch.min(loss1, loss2)
        # Returning the mean of all elements in the input tensor
        calculated_loss = torch.mean(calculated_loss)

        # Returning the calculated loss
        return calculated_loss


# =============================================================================
# END PpoAgent CLASS -- START run_ppo
# =============================================================================

def run_ppo(agent):
    """
    run_ppo

    Parameters
    ----------
    agent : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

# =============================================================================
# Initialization
# =============================================================================

    # Setting up our Atari Gym environment ()
    atari_env = gym.make(ppo_settings.ATARI_GAME)
    atari_env.reset()

    # Constructing an optimizerimizer
    # https://pytorch.org/docs/stable/optim.html
    optimizer = torch.optim.Adam(agent.parameters(),
                                 lr=ppo_settings.LEARNING_RATE)

    # Initializing running_average to None
    running_average = None

    # Initializing the log for recording our data for processing later
    reward_log = open('rewardlog.txt', 'a')

# =============================================================================
# Begin execution
# =============================================================================

    # for each iteration
    for it in range(ppo_settings.MAX_ITERATIONS):

        # Initializing lists
        state_history = list()
        action_history = list()
        probability_of_action_history = list()
        reward_history = list()

        # Each iteration will have 10 episodes
        for episode in range(ppo_settings.NUMBER_EPISODES):
            image_state, prev_image_state = atari_env.reset(), None

            for t in range(ppo_settings.NUM_T):
                # Rendering the Pong environment
                atari_env.render()

                # Preprocessing the image
                processed_image_state = agent.preprocess(image_state,
                                                          prev_image_state)

                # Disabling gradient calculation
                # https://pytorch.org/docs/stable/generated/torch.no_grad.html
                with torch.no_grad():
                    action, probability_of_action = agent(processed_image_state)

                prev_image_state = image_state
                image_state, reward, done, _ = atari_env.step(agent.get_next_step(action))

                # Appending to lists
                state_history.append(processed_image_state)
                action_history.append(action)
                probability_of_action_history.append(probability_of_action)
                reward_history.append(reward)

                if done:
                    reward_sum = sum(reward_history[-t:])
                    running_average = (0.99
                                       * running_average
                                       + 0.01
                                       * reward_sum if running_average else reward_sum)

                    # Writing the results to our log
                    reward_log.write(str(it) + '\t'
                                     + str(episode) + '\t'
                                     + str(t) + '\t'
                                     + str(action) + '\t'
                                     + str(probability_of_action) + '\t'
                                     + str(reward_sum) + '\t'
                                     + str(running_average) + '\n')

                    reward_log.flush()

                    break

# =============================================================================
# Calculating Advantage
# =============================================================================

        # Initializing
        R = 0
        d_rewards = list()

        for r in reward_history[::-1]:
            if r != 0:
                R = 0 # resetting

            R = r + agent.gamma * R
            d_rewards.insert(0, R)

        d_rewards = torch.FloatTensor(d_rewards)
        d_rewards = (d_rewards - d_rewards.mean()) / d_rewards.std()

        # Updating our policy
        for _ in range(ppo_settings.UPDATE_POLICY_RANGE):
            num_batch = ppo_settings.NUM_BATCH

            # Random number idx
            random_idx = random.sample(range(len(action_history)), num_batch)

            # Concatenate the given sequence of tensors in the given dimension
            # https://pytorch.org/docs/stable/generated/torch.cat.html
            state_batch = torch.\
                cat([state_history[current_idx] for current_idx in random_idx], 0)

            # Specifying torch.LongTensor type, a multi-dimensional matrix containing
            # elements of a single data type. Each type has CPU and GPU variants
            # https://pytorch.org/docs/stable/tensors.html
            action_batch = torch.\
                LongTensor([action_history[current_idx] for current_idx in random_idx])

            # Specifying torch.FloatTensor type
            probability_of_action_batch = torch.\
                FloatTensor([probability_of_action_history[current_idx] for current_idx in random_idx])
            advantage_batch = torch.\
                FloatTensor([d_rewards[current_idx] for current_idx in random_idx])

            optimizer.zero_grad()

            calculated_loss = agent(state_batch,
                                    action_batch,
                                    probability_of_action_batch,
                                    advantage_batch)

            calculated_loss.backward()
            # Stepping
            optimizer.step()

        # Saving Checkpoint
        if it % 5 == 0:
            torch.save(agent.state_dict(), 'params.ckpt')

    # Complete Execution
    atari_env.close()
