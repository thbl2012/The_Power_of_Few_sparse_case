import numpy as np
import math, sys, time, os, glob
from datetime import timedelta
from collections import Counter
from random_graph import RandomGraph

MAX_DAYS = 100
CYCLE_ONE = 1
CYCLE_TWO = 2
INCONCLUSIVE = 0
status_to_text = {
  CYCLE_ONE: 'Cycle one', CYCLE_TWO: 'Cycle_two', INCONCLUSIVE: 'Inconclusive'
}
stats_list = [
  'status',  # 0: inconclusive after MAX_DAYS days, 1: period of 1, 2: period of 2
  'r(t-1)',  # for period 2 status, number of Reds on the day before the first repeated day
  'b(t-1)',  # for period 2 status, number of Blues on the day before the first repeated day
  'r(t)',  # for period 2 status, number of Reds on the first repeated day
  'b(t)',  # for period 2 status, number of Blues on the first repeated day
  't',  # the first repeated day
]
info_to_index = dict()
for i, field in enumerate(stats_list):
  info_to_index[field] = i
DATA_DIR = 'optimal_power_of_few'
MAX_DIGITS = 6


def print_progress_bar(start_time, n_trials_done, n_trials_total, my_dict, key1, key2, key3):
  # print progress bar
  sys.stdout.write(
    '\rCompleted: {}/{} trials. {}: {}, {}: {}, {}: {}. Elapsed: {}'.format(
      n_trials_done, n_trials_total,
      status_to_text[key1], my_dict[key1],
      status_to_text[key2], my_dict[key2],
      status_to_text[key3], my_dict[key3],
      timedelta(seconds=time.time() - start_time)))
  sys.stdout.flush()


def trial(n, p, c):
  g = RandomGraph(n)
  g.set_color(c)
  g.generate(p)
  prev_colors = g.colors
  prev_two_colors = g.colors
  for day in range(1, MAX_DAYS + 1):
    g.transition()
    if np.array_equal(g.colors, prev_colors):
      return (CYCLE_ONE, ) + g.count(prev_colors) + g.count() + (day, )
    if day > 1 and np.array_equal(g.colors, prev_two_colors):
      return (CYCLE_TWO, ) + g.count(prev_colors) + g.count() + (day, )
    prev_two_colors = prev_colors
    prev_colors = g.colors
  return (INCONCLUSIVE, ) + g.count(prev_two_colors) + g.count() + (MAX_DAYS, )


def main(n=10000, f=1, delta=1, n_trials=1000, save=True):
  """
  Runs multiple trials, where each trial has 3 steps:
  - generate a random graph G from the Erdos-Renyi G(n, p) distribution, where p = f/delta^2
  - run Majority Dynamics on G, with the n/2 + delta vertices initially assigned Red
  - collect relevant statistics about the trials, as described in the variable stats_list
  :param n: number of vertices in G
  :param f: p * delta^2
  :param delta: initial advantage for Red
  :param n_trials: number of trials to run
  :param save: if True, the statistics in stats_list are saved for each trial and combined into a .npy file
  :return: None, but a summary is printed (even when save = False) for each status that contains
           the number of times and frequency that status occurs
  """
  p = f / delta ** 2
  print('Proceeding to run n_trials = {} trials on random graph G(n, p) '
        'for n = {}, p = {}/c^2 = {}.'.format(n_trials, n, f, p))
  print('Max days = {}. Advantage c = {}. Starting with {} Reds and {} Blues.'.format(
      MAX_DAYS, delta, math.ceil(n / 2 + delta), math.floor(n / 2 - delta)))
  print()

  # run n_trials trials
  records = []  # dict format: (status, n_red, n_blue): number of occurences
  summary = {CYCLE_ONE: 0, CYCLE_TWO: 0, INCONCLUSIVE: 0}
  start = time.time()
  for i in range(1, n_trials + 1):
    result = trial(n, p, delta)
    records.append(result)
    summary[result[0]] += 1
    print_progress_bar(start, i, n_trials, summary, CYCLE_ONE, CYCLE_TWO, INCONCLUSIVE)
  print()
  print()
  print('============== SUMMARY ==================')
  print('Status           Count     Frequency')
  for status, count in summary.items():
    print('{:<12}      {:<6}      {:.2f}'.format(
      status, count, count / n_trials
    ))

  # save records
  if save:
    data_subdir = '{}/{}_{}_{}'.format(DATA_DIR, n, str(f).replace('.', 'd'), str(delta).replace('.', 'd'))
    os.makedirs(data_subdir, exist_ok=True)
    print(data_subdir)
    batch_list = [int(fname[- MAX_DIGITS - 4:-4]) for fname in glob.glob(data_subdir + '/*.npy')]
    first_batch = max(batch_list) + 1 if batch_list else 0
    np.save('{}/{:0{}}.npy'.format(data_subdir, first_batch, MAX_DIGITS), records)


if __name__ == '__main__':
  main(n=10000, f=1, delta=10, n_trials=10, save=False)


