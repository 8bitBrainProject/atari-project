import sys
import random
from wrappers import make_atari, wrap_deepmind  #, wrap_pytorch

if __name__ == '__main__':
    render = True #!!!!!

    env = make_atari(sys.argv[1])
    num_actions = env.action_space.n
    env = wrap_deepmind(env)

    observation = env.reset()
    running_reward = None
    reward_sum = 0
    episode_number = 0  #!!!!! for picking back up where last series was stopped
    action = 0

    log_file_name = 'random ' + sys.argv[2] + '1e-4 250 no' + sys.argv[3] + '.txt'
    reward_log = open(log_file_name, 'w') #!!!!! added log file

    while (episode_number < 250):
      if render: env.render()

      action = random.randint(0, num_actions-1)
      # step the environment and get new measurements
      observation, reward, done, info = env.step(action)
      reward_sum += reward

      if done: # an episode finished
        episode_number += 1

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        #print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)) #!!!!! add parens to print
        print('episode:', episode_number, 'reward:', reward_sum, 'running reward:', running_reward)
        reward_log.write(str(episode_number) + '\t' + str(reward_sum) + '\t' + str(running_reward) + '\n') #!!!!! support logging
        reward_log.flush() #!!!!! support logging
        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = None

