# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy
import traceback

"""
timePlotterAvg.py: Plot the mean, min, max, and standard deviation of time
averaged over 30 runs.
"""

num_runs = 30
num_episodes = 500
run_episodes = numpy.zeros(num_episodes)
run_time_lists = numpy.zeros((num_episodes, num_runs))

for curr_run in range(num_runs):
    # filename = 'ppo-logs/ppo pv4w 1e-4 no' + str(curr_run) + '.txt'
    # filename = 'pg-logs/pg pv4w 1e-4 500 no' + str(curr_run) + '.txt'
    # filename = 'ddqn-logs/ddqn pv4w 1e-4 500 no' + str(curr_run) + '.txt'
    filename = 'random-logs/random pv4w 500 no' + str(curr_run) + '.txt'
    curr_episode = 0
    with open(filename, 'r') as reader:
        try:
            curr_line = reader.readline()
            while (curr_line):
                data_list = list(curr_line.split("\t"))
                if (filename.startswith('ppo')):
                    run_episodes[curr_episode] = int((int(data_list[0]) * 10) + int(data_list[1]) + 1)
                    run_time_lists[curr_episode][curr_run] = float(data_list[7])
                else:
                    run_episodes[curr_episode] = int(data_list[0])
                    run_time_lists[curr_episode][curr_run] = float(data_list[3])
                curr_line = reader.readline()
                curr_episode += 1
        except:
            print('Problem in line: |' + curr_line + '|')
            traceback.print_exc()
            pass

run_average_times = numpy.zeros(num_episodes)
run_max_times = numpy.zeros(num_episodes)
run_min_times = numpy.zeros(num_episodes)
run_std_times = numpy.zeros(num_episodes)

for epi in range(num_episodes):
    run_average_times[epi] = numpy.mean(run_time_lists[epi])
    run_max_times[epi] = numpy.amax(run_time_lists[epi])
    run_min_times[epi] = numpy.amin(run_time_lists[epi])
    run_std_times[epi] = numpy.std(run_time_lists[epi])

plt.errorbar(run_episodes, run_average_times,
             [run_average_times - run_min_times,
              run_max_times - run_average_times],
             lw = 0.3, fmt = 'g', label = 'Min & Max')
plt.errorbar(run_episodes, run_average_times, run_std_times,
             lw = 0.6, fmt = 'b', label = 'Std Dev')
plt.errorbar(run_episodes, run_average_times, fmt='k', label='Mean')

# plt.title('DDQN Cumulative Run Time over 30 Runs (mean, min, max, std)')
# plt.title('PPO Cumulative Run Time over 30 Runs (mean, min, max, std)')
# plt.title('PG Cumulative Run Time over 30 Runs (mean, min, max, std)')
plt.title('Random Cumulative Run Time over 30 Runs (mean, min, max, std)')
plt.xlabel('Episodes')
plt.ylabel('Time (s)')
plt.legend(loc='lower right')
plt.grid(True)

# plt.show()

# plt.savefig('plots/DDQN Cumulative Run Time over 30 Runs.png', dpi = 600)
# plt.savefig('plots/PPO Cumulative Run Time over 30 Runs.png', dpi = 600)
# plt.savefig('plots/PG Cumulative Run Time over 30 Runs.png', dpi = 600)
plt.savefig('plots/Random Cumulative Run Time over 30 Runs.png', dpi = 600)

# Ref:
# https://stackoverflow.com/questions/33328774/box-plot-with-min-max-average-and-standard-deviation


