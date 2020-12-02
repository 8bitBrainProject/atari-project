"""
COMP 5600/6600/6606 FINAL PROJECT

TEAM: 8-BIT BRAIN

@author: Dennis Brown (dgb0028@auburn.edu)
@author: Shannon McDade (slm0035@auburn.edu)
@author: Jacob Parmer (jdp0061@auburn.edu)

DESCRIPTION: PgAgent Agent Implementation for Atari 2600 Games
             Currently, will only run Pong
             Adapted from http://karpathy.github.io/2016/05/31/rl/
"""

import numpy as np
import time
from agents.wrappers import make_atari, wrap_deepmind


class PgAgent:
    """
    Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym.
    """
    def __init__(self):
        # hyperparameters
        self.H = 200 # number of hidden layer neurons
        self.batch_size = 10 # every how many episodes to do a param update?
        self.learning_rate = 1e-4
        self.gamma = 0.99 # discount factor for reward
        self.decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
        self.render = True
        self.start_time = time.time()

        # model initialization
        self.D = 84 * 84 # input dimensionality: 84x84 grid
        self.model = {}
        self.model['W1'] = np.random.randn(self.H,self.D) / np.sqrt(self.D) # "Xavier" initialization
        self.model['W2'] = np.random.randn(self.H) / np.sqrt(self.H)

        self.grad_buffer = { k : np.zeros_like(v) for k,v in self.model.items() } # update buffers that add up gradients over a batch
        self.rmsprop_cache = { k : np.zeros_like(v) for k,v in self.model.items() } # rmsprop memory

        self.epx = None


    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]


    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def policy_forward(self, x):
        h = np.dot(self.model['W1'], x)
        h[h<0] = 0 # ReLU nonlinearity
        logp = np.dot(self.model['W2'], h)
        p = self.sigmoid(logp)
        return p, h # return probability of taking action 2, and hidden state


    def policy_backward(self, eph, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.model['W2'])
        dh[eph <= 0] = 0 # backpro prelu
        dW1 = np.dot(dh.T, self.epx)
        return {'W1':dW1, 'W2':dW2}


    def run_pg(self):
        start_time = time.time()

        env = make_atari("PongNoFrameskip-v4")
        env = wrap_deepmind(env)
        observation = env.reset()
        prev_x = None # used in computing the difference frame
        xs,hs,dlogps,drs = [],[],[],[]
        running_reward = None
        reward_sum = 0
        episode_number = 0

        reward_log = open('rewardlog-pg.txt', 'w')

        while (True):  # Run until user kills process
            if self.render: env.render()

            cur_x = observation.astype(np.float).ravel()
            x = cur_x - prev_x if prev_x is not None else np.zeros(self.D)
            prev_x = cur_x

            # forward the policy network and sample an action from the returned probability
            aprob, h = self.policy_forward(x)
            action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

            # record various intermediates (needed later for backprop)
            xs.append(x) # observation
            hs.append(h) # hidden state
            y = 1 if action == 2 else 0 # a "fake label"
            dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

            # step the environment and get new measurements
            observation, reward, done, info = env.step(action)
            reward_sum += reward

            drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

            if done: # an episode finished
                episode_number += 1

                # stack together all inputs, hidden states, action gradients, and rewards for this episode
                self.epx = np.vstack(xs)
                eph = np.vstack(hs)
                epdlogp = np.vstack(dlogps)
                epr = np.vstack(drs)
                xs,hs,dlogps,drs = [],[],[],[] # reset array memory

                # compute the discounted reward backwards through time
                discounted_epr = self.discount_rewards(epr)

                # standardize the rewards to be unit normal (helps control the gradient estimator variance)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)

                epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
                grad = self.policy_backward(eph, epdlogp)
                for k in self.model: self.grad_buffer[k] += grad[k] # accumulate grad over batch

                # perform rmsprop parameter update every batch_size episodes
                if episode_number % self.batch_size == 0:
                    for k,v in self.model.items():
                        g = self.grad_buffer[k] # gradient
                        self.rmsprop_cache[k] = self.decay_rate * self.rmsprop_cache[k] + (1 - self.decay_rate) * g**2
                        self.model[k] += self.learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
                        self.grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

                # boring book-keeping
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print('episode:', episode_number, 'reward:', reward_sum, 'running reward:', running_reward)
                reward_log.write(str(episode_number) + '\t' + str(reward_sum) + '\t' + str(running_reward) + '\t' + str(time.time() - start_time) + '\n')
                reward_log.flush()
                reward_sum = 0

                observation = env.reset() # reset env
                prev_x = None

