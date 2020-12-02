"""
COMP 5600/6600/6606 FINAL PROJECT

TEAM: 8-BIT BRAIN

@author: Dennis Brown (dgb0028@auburn.edu)
@author: Shannon McDade (slm0035@auburn.edu)
@author: Jacob Parmer (jdp0061@auburn.edu)

DESCRIPTION: PPO Settings

RESOURCES:
    > This agent was inspired by and modeled after the PPO agent of Sagar Gubbi
        Source: https://www.sagargv.com/blog/pong-ppo/

"""

# =============================================================================
# PRIMARY SETTINGS (adjust these as needed, defaults provided)
# =============================================================================

ATARI_GAME = 'PongNoFrameskip-v4'

# eps clip
EPISODE_CLIP = 0.1

# Discount Factor (Gamma)
GAMMA = 0.99

# Learning rate
LEARNING_RATE = 1e-3

# =============================================================================
# Not recommended to update anything below here
# =============================================================================

# Step Size
STEP_SIZE = 2

MAX_ITERATIONS = 100000

NUMBER_EPISODES = 10

NUM_T = 190000

NUM_BATCH = 27000

UPDATE_POLICY_RANGE = 5

# =============================================================================
# Sequential Container Settings (not recommended to adjust these)
# =============================================================================

# In_features: size of each input sample for nn.Linear(in_features, out_features)
IN_FEATURES1 = 6000

# Out_features: size of each output sample for nn.Linear(in_features, out_features)
OUT_FEATURES1 = 512

# In_features: size of each input sample for nn.Linear(in_features, out_features)
IN_FEATURES2 = 512

# Out_features: size of each output sample for nn.Linear(in_features, out_features)
OUT_FEATURES2 = 2

# =============================================================================
# Image Processing Settings (not recommended to adjust these)
# =============================================================================

# Pixels to crop from image
CROP_BEGINNING = 35
CROP_END = 185

# Downsampling factor
DOWNSAMPLE_FACTOR = 2

# Background Erase Settings
BACKGROUND_ERASE1 = 144
BACKGROUND_ERASE2 = 109
