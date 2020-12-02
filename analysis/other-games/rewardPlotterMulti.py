# -*- coding: utf-8 -*-
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

"""
rewardPlotterMulti.py: Plot the running mean of reward vs. episode for 5 runs.
"""


alg = 'ddqn'
gameenv = 'mpv4w'
algname = 'DDQN'
gamename = 'Ms. Pac-Man'
num_runs = 1

datafiles = []
for i in range(num_runs):
    filename = alg + '-logs/' + alg + ' ' + gameenv + '1e-4 2500 no' + str(i) + '.txt'
    datafiles.append(filename)

data = [[] for _ in range(len(datafiles))]

max_eps = 2500

for curr_file in range(len(datafiles)):
    data[curr_file] = numpy.loadtxt(datafiles[curr_file])
    data_label = (datafiles[curr_file].split('.'))[0]

    # Raw data
    eps = None
    scores = None
    if (datafiles[curr_file].startswith('ppo')):
        eps = (data[curr_file][:, 0] * 10) + data[curr_file][:, 1] + 1
        scores = data[curr_file][:, 5]
    else:
        eps = data[curr_file][:, 0]
        scores = data[curr_file][:, 1]
    #plt.plot(eps, scores)

    if (len(eps) > max_eps):
        eps = eps[0:(max_eps - 1)]
        scores = scores[0:(max_eps - 1)]

    # Rolling average
    # avgscores = numpy.convolve(scores, numpy.ones(25), 'valid') / 25
    # avgeps = eps[0:len(avgscores)]
    # plt.plot(avgeps, avgscores, label = data_label)

    # # Literal mean -- useless
    # mean = numpy.zeros(len(scores))
    # total = 0
    # for i in range(0, len(scores)):
    #     total += scores[i]
    #     mean[i] = total / (i + 1)
    # plt.plot(eps, mean, label = data_label)

    # Running mean
    runningmean = numpy.zeros(len(scores))
    runningmean[0] = scores[0]
    for i in range(1, len(scores)):
        runningmean[i] = runningmean[i - 1] * 0.99 + scores[i] * 0.01

    plt.plot(eps, runningmean, label = data_label)

    # Fit exponential function
    # pars, cov = curve_fit(f=exponential, xdata=eps, ydata=scores, p0=[0, 0], bounds=(-numpy.inf, numpy.inf))
    # print(pars)
    # plt.plot(eps, exponential(eps, *pars))

    # # Predict the future
    # future_eps = numpy.linspace(start = numpy.max(eps), stop = 2000)
    # plt.plot(future_eps, exponential(future_eps, *pars))

# plt.legend(loc='lower right')
# plt.title('Reward over 30 DDQN Runs')
# plt.title('Reward over 30 PPO Runs')
# plt.title('Reward over 30 PG Runs')
plt.title(algname + '-' + gamename + ' Reward over ' + str(num_runs) + ' Runs')
plt.xlabel('Episodes')
plt.ylabel('Running Average Reward')
# plt.ylim(-21, 21)
plt.grid(True)
#plt.show()

plt.savefig('plots/' + gamename + ' Reward over ' + str(num_runs) + ' ' + algname + ' Runs.png', dpi = 600)