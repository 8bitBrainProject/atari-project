# -*- coding: utf-8 -*-
import numpy
import scipy.stats as stats

"""
makeTheTable.py: Compile data from all runs into a LaTeX table
"""


def read_files(file_prefix, num_files):
    """
    Read the results of the log files and return a list of the best of each run
    (we only care about the last line of each file).
    """

    data = []
    for curr_file in range(num_files):
        filename = file_prefix + str(curr_file) + '.txt'
        curr_data = [0, 0, 0]
        with open(filename, 'r') as reader:
           curr_line = reader.readline()
           while (curr_line):
                data_list = list(curr_line.split("\t"))
                if (filename.startswith('ppo')):
                    curr_data[0] = float(data_list[6])
                    curr_data[1] = 21 - float(data_list[6])
                    curr_data[2] = float(data_list[7])
                else:
                    curr_data[0] = float(data_list[2])
                    curr_data[1] = 21 - float(data_list[2])
                    curr_data[2] = float(data_list[3])
                curr_line = reader.readline()
        data.append(curr_data)
    return data


# Compare two log files

title1 = 'Random'
title2 = 'PG'
title3 = 'PPO'
title4 = 'DDQN'

data1 = read_files('random-logs/random pv4w 500 no', 30)
data2 = read_files('pg-logs/pg pv4w 1e-4 500 no', 30)
data3 = read_files('ppo-logs/ppo pv4w 1e-4 no', 30)
data4 = read_files('ddqn-logs/ddqn pv4w 1e-4 500 no', 30)
runs = numpy.linspace(1, len(data1), num = len(data1), endpoint=True)

for i in range(30):
    print(int(runs[i]), '&', end = ' ')
    print("%.2f &" % round(data1[i][0], 2), end = ' ')
    print("%.2f &" % round(data1[i][1], 2), end = ' ')
    print("%.2f &" % round(data1[i][2], 2), end = ' ')
    print("%.2f &" % round(data2[i][0], 2), end = ' ')
    print("%.2f &" % round(data2[i][1], 2), end = ' ')
    print("%.2f &" % round(data2[i][2], 2), end = ' ')
    print("%.2f &" % round(data3[i][0], 2), end = ' ')
    print("%.2f &" % round(data3[i][1], 2), end = ' ')
    print("%.2f &" % round(data3[i][2], 2), end = ' ')
    print("%.2f &" % round(data4[i][0], 2), end = ' ')
    print("%.2f &" % round(data4[i][1], 2), end = ' ')
    print("%.2f \\\\ \\hline" % round(data4[i][2], 2))
