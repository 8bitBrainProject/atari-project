# atari-project

Run the reinforcement learning algorithm agents Policy Gradient, Proximal Policy Optimization, Deep Q-Learning, Double Deep Q Learning, and the non-learning Random agent on Pong. These agents are described in more detail in the associated Final Report.

To choose an agent, edit the last line of main.py to set the algorithm you want to run.

Then run:
```
python main.py
```

NOTE: To run any of these algorithms you need OpenAI Gym and Arcade Learning Environment installed. See the last Appendix of the Final Report for instructions.

## agents directory

Modules containing the actual agents: PG, PPO, DQN, DDQN, and Random, plus a couple of support modules.

## models directory

QNetwork model for DQN.

NOTE: Models for PG, PPO, and DDQN are in the agent modules themselves.

## config directory

Edit the settings module associated with the agent you want to run (PPO and DQN).

NOTE: Settings for PG, DDQN, and Random are in the agent modules themselves.

## analysis directory

The analysis directory holds data and code supporting the exploratory and formal analysis of PG, PPO, DDQN, and Random algorithms on Atari Pong with optimal settings (game = PongNoFrameskip-v4, learning rate = 1e-4).

See the README in the analysis directory for more details.

