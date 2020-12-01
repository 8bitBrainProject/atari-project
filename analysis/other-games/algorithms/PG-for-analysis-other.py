""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import sys
import numpy as np
import random
import pickle as pickle
from wrappers import make_atari, wrap_deepmind  #, wrap_pytorch



def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  # """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  # I = I[35:195] # crop
  # I = I[::2,::2,0] # downsample by factor of 2
  # I[I == 144] = 0 # erase background (background type 1)
  # I[I == 109] = 0 # erase background (background type 2)
  # I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  # dW2 = np.dot(eph.T, epdlogp).ravel()
  dW2 = np.dot(eph.T, epdlogp).T
  # dh = np.outer(epdlogp, model['W2'])
  dh = np.dot(epdlogp, model['W2'])    # !!!!!!
  dh[eph <= 0] = 0 # backprop relu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}


if __name__ == '__main__':
    # hyperparameters
    H = 200 # number of hidden layer neurons
    batch_size = 10 # every how many episodes to do a param update?
    learning_rate = 1e-4
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
    resume = False # resume from previous checkpoint?  #!!!!!
    render = True #!!!!!

    env = make_atari(sys.argv[1])
    num_actions = env.action_space.n
    env = wrap_deepmind(env)

    # model initialization
    D = 84 * 84 # input dimensionality: 80x80 grid
    if resume:
      print('RESUMING') #!!!!!
      model = pickle.load(open('save.p', 'rb'))
    else:
      model = {}
      model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
      model['W2'] = np.random.randn(num_actions, H) / np.sqrt(H)

    grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch #!!!!! iteritems -> items
    rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory #!!!!! iteritems -> items

    observation = env.reset()
    prev_x = None # used in computing the difference frame
    xs,hs,dlogps,drs = [],[],[],[]
    running_reward = None
    reward_sum = 0
    episode_number = 0  #!!!!! for picking back up where last series was stopped
    action = 0

    log_file_name = 'pg ' + sys.argv[2] + '1e-4 250 no' + sys.argv[3] + '.txt'
    reward_log = open(log_file_name, 'w') #!!!!! added log file

    while (episode_number < 250):
      if render: env.render()

      # preprocess the observation, set input to network to be difference image
      cur_x = prepro(observation)
      x = cur_x - prev_x if prev_x is not None else np.zeros(D)
      prev_x = cur_x

      # forward the policy network and sample an action from the returned probability
      # aprob, h = policy_forward((np.vstack([x] * num_actions)).T)
      # action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
      aprob, h = policy_forward(x)
      if ((np.sum(aprob) != 0)
          and (not (np.isnan(np.sum(aprob))))):
          action = np.random.choice(num_actions, p = (aprob / np.sum(aprob)))
      else:
          action = random.randint(0, num_actions-1)



      # record various intermediates (needed later for backprop)
      xs.append(x) # observation
      hs.append(h) # hidden state
      y = np.zeros(num_actions)
      y[action] = 1
      dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

      # step the environment and get new measurements
      observation, reward, done, info = env.step(action)
      reward_sum += reward

      drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

      if done: # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp)
        for k in model:
            grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
          for k,v in model.items():
            g = grad_buffer[k] # gradient
            rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
            model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
            grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        #print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)) #!!!!! add parens to print
        print('episode:', episode_number, 'reward:', reward_sum, 'running reward:', running_reward)
        reward_log.write(str(episode_number) + '\t' + str(reward_sum) + '\t' + str(running_reward) + '\n') #!!!!! support logging
        reward_log.flush() #!!!!! support logging
        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = None

