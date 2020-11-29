# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy
import traceback

"""
errorPlotterAvg.py: Plot the mean, min, max, and standard deviation of error
averaged over 30 runs.
"""

num_runs = 30
num_episodes = 500
run_episodes = numpy.zeros(num_episodes)
run_error_lists = numpy.zeros((num_episodes, num_runs))

for curr_run in range(num_runs):
    # filename = 'ppo-logs/ppo pv4w 1e-4 no' + str(curr_run) + '.txt'
    filename = 'pg-logs/pg pv4w 1e-4 500 no' + str(curr_run) + '.txt'
    # filename = 'ddqn-logs/ddqn pv4w 1e-4 500 no' + str(curr_run) + '.txt'
    curr_episode = 0
    with open(filename, 'r') as reader:
        try:
            curr_line = reader.readline()
            while (curr_line):
                data_list = list(curr_line.split("\t"))
                if (filename.startswith('ppo')):
                    run_episodes[curr_episode] = int((int(data_list[0]) * 10) + int(data_list[1]) + 1)
                    run_error_lists[curr_episode][curr_run] = float(data_list[5]) * -1.0
                else:
                    run_episodes[curr_episode] = int(data_list[0])
                    run_error_lists[curr_episode][curr_run] = float(data_list[1]) * -1.0
                curr_line = reader.readline()
                curr_episode += 1
        except:
            print('Problem in line: |' + curr_line + '|')
            traceback.print_exc()
            pass

run_average_errors = numpy.zeros(num_episodes)
run_max_errors = numpy.zeros(num_episodes)
run_min_errors = numpy.zeros(num_episodes)
run_std_errors = numpy.zeros(num_episodes)

for epi in range(num_episodes):
    run_average_errors[epi] = numpy.mean(run_error_lists[epi])
    run_max_errors[epi] = numpy.amax(run_error_lists[epi])
    run_min_errors[epi] = numpy.amin(run_error_lists[epi])
    run_std_errors[epi] = numpy.std(run_error_lists[epi])

plt.errorbar(run_episodes, run_average_errors,
             [run_average_errors - run_min_errors,
              run_max_errors - run_average_errors],
             lw = 0.3, fmt = 'g', label = 'Min & Max')
plt.errorbar(run_episodes, run_average_errors, run_std_errors,
             lw = 0.6, fmt = 'b', label = 'Std Dev')
plt.errorbar(run_episodes, run_average_errors, fmt='k', label='Mean')

# plt.title('DDQN Error over 30 Runs (mean, min, max, std)')
# plt.title('PPO Error over 30 Runs (mean, min, max, std)')
plt.title('PG Error over 30 Runs (mean, min, max, std)')
plt.xlabel('Episodes')
plt.ylabel('Error')
plt.legend(loc='upper right')
plt.grid(True)

# plt.show()

# plt.savefig('plots/DDQN Error over 30 Runs.png', dpi = 600)
# plt.savefig('plots/PPO Error over 30 Runs.png', dpi = 600)
plt.savefig('plots/PG Error over 30 Runs.png', dpi = 600)

# Ref:
# https://stackoverflow.com/questions/33328774/box-plot-with-min-max-average-and-standard-deviation


