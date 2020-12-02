# -*- coding: utf-8 -*-
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

"""
timePlotterMulti.py: Plot the run time vs. episode for 30 runs.
"""

# Function to calculate the exponential with constants a and b
def exponential(x, a, b):
    return -21 + a*numpy.exp(b*x)

# datafiles = [
#              # 'pg pv0 1e-3 20000.txt',
#              # 'pg pv0 1e-4 30400.txt',
#              # 'pg pv4 1e-3 3592.txt',
#              # 'pg pv4 1e-4 5079.txt',
#              # 'pg pv4w 1e-3 2634.txt',
#              # 'pg pv4w 1e-4 601.txt',
#              # 'pg pv4w 1e-4 1030 no2.txt',
#              # 'pg pv4w 1e-4 1982 no3.txt',
#              # 'pg pv4w 1e-4 905 no4.txt'
#              # 'ppo pv0 1e-3 1709.txt',
#              # 'ppo pv0 1e-4 6497.txt',
#              # 'ppo pv4 1e-3 2048.txt',
#              # 'ppo pv4w 1e-3 829.txt',
#              'ppo pv4w 1e-4 5159.txt',
#              'ppo pv4w 1e-4 2682-no2.txt'
#              # 'ddqn pv0 1e-3 314.txt',
#              # 'ddqn pv0 1e-4 660.txt'
#              # 'ddqn pv4w 1e-3 546.txt',
#              # 'ddqn pv4w 1e-4 1152.txt'
#              # 'ddqnfail1 pv0 1e-4 652.txt'
#              ]

# datafiles = ['ddqn siv4w 1e-4 8108.txt',
#              'ddqn mpv4w 1e-4 3759.txt']

datafiles = []
for i in range(30):
    # datafiles.append('ppo-logs/ppo pv4w 1e-4 no' + str(i) + '.txt')
    # datafiles.append('pg-logs/pg pv4w 1e-4 500 no' + str(i) + '.txt')
    # datafiles.append('ddqn-logs/ddqn pv4w 1e-4 500 no' + str(i) + '.txt')
    datafiles.append('random-logs/random pv4w 500 no' + str(i) + '.txt')

data = [[] for _ in range(len(datafiles))]

max_eps = 500

for curr_file in range(len(datafiles)):
    data[curr_file] = numpy.loadtxt(datafiles[curr_file])
    data_label = (datafiles[curr_file].split('.'))[0]

    # Raw data
    eps = None
    times = None
    if (datafiles[curr_file].startswith('ppo')):
        eps = (data[curr_file][:, 0] * 10) + data[curr_file][:, 1] + 1
        times = data[curr_file][:, 7]
    else:
        eps = data[curr_file][:, 0]
        times = data[curr_file][:, 3]
    #plt.plot(eps, times)

    if (len(eps) > max_eps):
        eps = eps[0:(max_eps - 1)]
        times = times[0:(max_eps - 1)]

    # Rolling average
    avgtimes = numpy.convolve(times, numpy.ones(25), 'valid') / 25
    avgeps = eps[0:len(avgtimes)]
    # plt.plot(avgeps, avgtimes, label = data_label)

    # # Literal mean -- useless
    # mean = numpy.zeros(len(times))
    # total = 0
    # for i in range(0, len(times)):
    #     total += times[i]
    #     mean[i] = total / (i + 1)
    # plt.plot(eps, mean, label = data_label)

    # Running mean
    runningmean = numpy.zeros(len(times))
    runningmean[0] = times[0]
    for i in range(1, len(times)):
        runningmean[i] = runningmean[i - 1] * 0.99 + times[i] * 0.01

    plt.plot(eps, runningmean, label = data_label)

    # Fit exponential function
    # pars, cov = curve_fit(f=exponential, xdata=eps, ydata=times, p0=[0, 0], bounds=(-numpy.inf, numpy.inf))
    # print(pars)
    # plt.plot(eps, exponential(eps, *pars))

    # # Predict the future
    # future_eps = numpy.linspace(start = numpy.max(eps), stop = 2000)
    # plt.plot(future_eps, exponential(future_eps, *pars))

# plt.legend(loc='lower right')
# plt.title('Time over 30 DDQN Runs')
# plt.title('Time over 30 PPO Runs')
# plt.title('Time over 30 PG Runs')
plt.title('Time over 30 Random Runs')
plt.xlabel('Episodes')
plt.ylabel('Time (s)')
# plt.ylim(-21, 21)
plt.grid(True)
#plt.show()

# plt.savefig('plots/Time over 30 DDQN Runs.png', dpi = 600)
# plt.savefig('plots/Time over 30 PPO Runs.png', dpi = 600)
# plt.savefig('plots/Time over 30 PG Runs.png', dpi = 600)
plt.savefig('plots/Time over 30 Random Runs.png', dpi = 600)
