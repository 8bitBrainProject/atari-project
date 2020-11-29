# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy
import scipy.stats as stats


"""
timeCompare.py: Perform F- and t-test on two data sets; plot comparative histogram of data.
"""


def read_files(file_prefix, num_files):
    """
    Read the results of the log file and return a list of the best of each run.
    """

    last_runtime = -1
    runtimes = []
    for curr_file in range(num_files):
        filename = file_prefix + str(curr_file) + '.txt'

        with open(filename, 'r') as reader:
           curr_line = reader.readline()
           while (curr_line):
                data_list = list(curr_line.split("\t"))
                if (filename.startswith('ppo')):
                    last_runtime = float(data_list[7]) # time
                else:
                    last_runtime = float(data_list[3]) # time
                curr_line = reader.readline()

        runtimes.append(last_runtime)

    return runtimes


# Compare two log files

# title1 = 'PG'
title1 = 'PPO'
title2 = 'DDQN'
combo_fileroot = title1 + '-' + title2
aspect = '-Time'

runtimes2 = read_files('ddqn-logs/ddqn pv4w 1e-4 500 no', 30)
runtimes1 = read_files('ppo-logs/ppo pv4w 1e-4 no', 30)
# runtimes1 = read_files('pg-logs/pg pv4w 1e-4 500 no', 30)
runs = numpy.linspace(1, len(runtimes1), num = len(runtimes1), endpoint=True)

# Calculate F-Test Two-Sample for Variances
mean1 = numpy.mean(runtimes1)
mean2 = numpy.mean(runtimes2)
var1 = numpy.var(runtimes1, ddof=1)
var2 = numpy.var(runtimes2, ddof=1)
obs = len(runtimes1)
ft_df = obs - 1
f = var1/var2
ft_p = stats.f.cdf(f, ft_df, ft_df)
alpha = 0.05
fcrit = stats.f.ppf(alpha, ft_df, ft_df)

have_equal_variances = False

print('-----------------------------')
print('\\begin{figure}[H]')
print('\\caption{' + title1 + ' vs. ' + title2 + ' -- Best values over ' + str(obs) + ' runs}')
print('\\centering')
print('\\includegraphics[width=8cm]{' + combo_fileroot + '.png}')
print('\\label{fig:' + combo_fileroot + aspect + '}')
print('\\end{figure}')
print()
print('\\begin{table}[H]')
print('\\centering')
print('\\caption{F-Test for ' + title1 + ' vs. ' + title2 + ' with $\\alpha = ' + str(alpha) + '$}')
print('\\label{tab:ftest-' + combo_fileroot + aspect + '}')
print('\\begin{tabular}{lll}')
print('\\hline')
print(' & ' + title1 + ' & ' + title2 + ' \\\\ \\hline')
print('\\multicolumn{1}{|l|}{Mean}     & \\multicolumn{1}{l|}{' + str(mean1) + '} & \\multicolumn{1}{l|}{' + str(mean2) + '} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{Variance}     & \\multicolumn{1}{l|}{' + str(var1) + '} & \\multicolumn{1}{l|}{' + str(var2) + '} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{Observations}     & \\multicolumn{1}{l|}{' + str(obs) + '} & \\multicolumn{1}{l|}{' + str(obs) + '} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{df}     & \\multicolumn{1}{l|}{' + str(ft_df) + '} & \\multicolumn{1}{l|}{' + str(ft_df) + '} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{F}     & \\multicolumn{1}{l|}{' + str(f) + '} & \\multicolumn{1}{l|}{} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{P(F$\leq$f) one-tail}     & \\multicolumn{1}{l|}{' + str(ft_p) + '} & \\multicolumn{1}{l|}{} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{F Critical one-tail}     & \\multicolumn{1}{l|}{' + str(fcrit) + '} & \\multicolumn{1}{l|}{} \\\\ \\hline')
print('\\end{tabular}')
print('\\end{table}')
print()

if (abs(mean1) > abs(mean2)) and (f < fcrit):
    print('\\noindent abs(mean 1) $>$ abs(mean 2) and F $<$ F Critical implies equal variances.')
    have_equal_variances = True
if (abs(mean1) > abs(mean2)) and (f > fcrit):
    print('\\noindent abs(mean 1) $>$ abs(mean 2) and F $>$ F Critical implies unequal variances.')
    have_equal_variances = False
if (abs(mean1) < abs(mean2)) and (f > fcrit):
    print('\\noindent abs(mean 1) $<$ abs(mean 2) and F $>$ F Critical implies equal variances.')
    have_equal_variances = True
if (abs(mean1) < abs(mean2)) and (f < fcrit):
    print('\\noindent abs(mean 1) $<$ abs(mean 2) and F $<$ F Critical implies unequal variances.')
    have_equal_variances = False
print()

# Calculate T-Test Two-Sample for equal or unequal variances
tt_df = (obs * 2) - 2
tcrit_two_tail = stats.t.ppf(1.0 - (alpha/2), tt_df)
(tstat, tt_p_two_tail) = stats.ttest_ind(runtimes1, runtimes2, equal_var=have_equal_variances)

print('\\begin{table}[H]')
print('\\centering')
print('\\caption{t-Test for ' + title1 + ' vs. ' + title2 + ' with ')
if (have_equal_variances):
    print('Equal Variances}')
else:
    print('Unequal Variances}')
print('\\label{tab:ttest-' + combo_fileroot + aspect + '}')
print('\\begin{tabular}{lll}')
print('\\hline')
print(' & ' + title1 + ' & ' + title2 + ' \\\\ \\hline')
print('\\multicolumn{1}{|l|}{Mean}     & \\multicolumn{1}{l|}{' + str(mean1) + '} & \\multicolumn{1}{l|}{' + str(mean2) + '} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{Variance}     & \\multicolumn{1}{l|}{' + str(var1) + '} & \\multicolumn{1}{l|}{' + str(var2) + '} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{Observations}     & \\multicolumn{1}{l|}{' + str(obs) + '} & \\multicolumn{1}{l|}{' + str(obs) + '} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{df}     & \\multicolumn{1}{l|}{' + str(tt_df) + '} & \\multicolumn{1}{l|}{' + str(ft_df) + '} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{t Stat}     & \\multicolumn{1}{l|}{' + str(tstat) + '} & \\multicolumn{1}{l|}{} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{P(T$\leq$t) two-tail}     & \\multicolumn{1}{l|}{' + str(tt_p_two_tail) + '} & \\multicolumn{1}{l|}{} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{t Critical two-tail}     & \\multicolumn{1}{l|}{' + str(tcrit_two_tail) + '} & \\multicolumn{1}{l|}{} \\\\ \\hline')
print('\\end{tabular}')
print('\\end{table}')
print()

if (abs(tstat) > abs(tcrit_two_tail)):
    print('\\noindent abs(t Stat) $>$ abs(t Critical two-tail) so we reject the null hypothesis -- the two samples are statistically different.')
    print('The average improvement of ' + title1 + ' over ' + title2 + ' is ' + str(mean1 - mean2) + '.')
else:
    print('\\noindent abs(t Stat) $<$ abs(t Critical two-tail) so we accept the null hypothesis -- the two samples are NOT statistically different.')
print('-----------------------------')

# # Dump the data to CSV for Excel
# writer = open('data/' + combo_fileroot + '.csv', 'w')
# for i in range(len(runtimes1)):
#     writer.write(str(runtimes1[i]) + ', ' + str(runtimes2[i]) + '\n');
# writer.close()

# Plot the data
overall_max = numpy.max([numpy.max(runtimes1), numpy.max(runtimes2)])
overall_min = numpy.min([numpy.min(runtimes1), numpy.min(runtimes2)])
bins = numpy.arange(overall_min, overall_max + 1, (overall_max / 10.0))
plt.hist([runtimes1, runtimes2], bins, label = [title1, title2])

plt.title('Run times to 500 Episodes: ' + title1 + ' and ' + title2)
plt.xlabel('Run Time (s)')
plt.ylabel('Number of runs')
plt.legend(loc='upper left')
plt.grid(True)

plt.savefig('plots/' + combo_fileroot + aspect + '.png', dpi = 600)


