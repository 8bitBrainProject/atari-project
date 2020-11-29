# What's in this directory

This directory holds data and code supporting the exploratory and formal analysis of PG, PPO, and DDQN algorithms on Atari Pong with optimal settings (game = PongNoFrameskip-v4, learning rate = 1e-4).

## Algorithms

The versions of PG, PPO, and DDQN used to obtain the data used in the analysis are each "snapshotted" here in the "algorithms" folder. These are each very rough versions of each algorithm that we started running as soon as they were correct and complete, but not well-organized or well-documented, in the interest of maximizing our time. They are captured here to ensure repeatability of the experiment. The final cleaned-up version of each algorithm, which should produce the same results, can be found back at the top level of this repository.

To run any of these algorithms you need OpenAI Gym and Arcade Learning Environment installed.

* To run PG:
```
python PG-for-analysis.py <base filename for log>
```
* To run PPO:
```
python PPO-for-analysis.py <base filename for log>
```
* To run DDQN:
```
python DDQN-for-analysis.py <base filename for log>
```

## Data

Raw data logs from the 30 runs of each algorithm are in the "pg-logs," "ppo-logs," and "ddqn-logs" directories.

## Code for Plotting and Analysis

The Python modules (.py files) read the runs and plot various aspects and perform statistical analysis. As these modules are essentially quick-and-dirty utility scripts, each module is hardcoded to read a certain directory of data, so the file has to be manually changed to plot and/or analyze other data sets. In most cases this is simply uncommenting/commenting out code already written in the module.

* errorPlotterMulti.py: Plot the running mean of error vs. episode for 30 runs.
* rewardPlotterMulti.py: Plot the running mean of reward vs. episode for 30 runs.
* timePlotterMulti.py: Plot the running mean of cumulative run time vs. episode for 30 runs.
* errorPlotterAvg.py: Plot the mean, min, max, and standard deviation of error averaged over 30 runs.
* rewardPlotterAvg.py: Plot the mean, min, max, and standard deviation of reward averaged over 30 runs.
* timePlotterAvg.py: Plot the mean, min, max, and standard deviation of cumulative running time averaged over 30 runs.
* errorCompare.py: Perform F- and t-test on two error data sets; plot comparative histogram of data.
* rewardCompare.py: Perform F- and t-test on two reward data sets; plot comparative histogram of data.
* timeCompare.py: Perform F- and t-test on two time data sets; plot comparative histogram of data.

## Plots

Generated plots are stored in the "plots" directory.

# Exploration

The "exploration" directory holds artifacts from an informal ad-hoc exploration of algorithms and parameters. The reward data from various runs are in its "logs" directory, plots in its "plots" directory, and the code "rewardPlotterMulti.py" plots arbitrary reward vs. episode log files.
