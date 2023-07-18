import numpy as np
import math, sys, time, os, glob
from datetime import timedelta
from collections import Counter
import warnings
from random_graph import RandomGraph

MAX_DAYS = 100
CYCLE_ONE = 1
CYCLE_TWO = 2
COLOR_RED = 1
COLOR_BLUE = -1
INCONCLUSIVE = 0
status_to_text = {
  CYCLE_ONE: 'Cycle 1', CYCLE_TWO: 'Cycle 2', INCONCLUSIVE: 'Inconclusive'
}
stats_list = [
  'status',  # 0: inconclusive after MAX_DAYS days, 1: period of 1, 2: period of 2
  'r(t-1)',  # for period 2 status, number of Reds on the day before the first repeated day
  'b(t-1)',  # for period 2 status, number of Blues on the day before the first repeated day
  'r(t)',  # for period 2 status, number of Reds on the first repeated day
  'b(t)',  # for period 2 status, number of Blues on the first repeated day
  't',  # the first repeated day
  'iso_r',  # number of isolated Reds (not affected throughout the majority dynamics)
  'iso_b',  # number of isolated Blues (not affected throughout the majority dynamics)
  'r->b',  # number of Reds on day t that turn Blue on day t+1
  'b->r'  # number of Blues on day t that turn Red on day t+1
]

info_to_index = dict()
for i, field in enumerate(stats_list):
  info_to_index[field] = i

DATA_DIR = 'balanced_initial_colors'
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

  # Find isolated vertices
  is_iso = ~np.any(g.true_edges, axis=0)
  count_iso_r = np.count_nonzero(np.logical_and(is_iso, g.colors == COLOR_RED))
  count_iso_b = np.count_nonzero(np.logical_and(is_iso, g.colors == COLOR_BLUE))

  # Run the process
  prev_colors = g.colors
  prev_two_colors = g.colors
  for day in range(1, MAX_DAYS + 1):
    g.transition()
    if np.array_equal(g.colors, prev_colors):
      return (CYCLE_ONE, ) + g.count(prev_colors) + g.count() + (day, count_iso_r, count_iso_b, 0, 0)
    if day > 1 and np.array_equal(g.colors, prev_two_colors):
      r_to_b = np.count_nonzero(np.logical_and(
        g.colors == COLOR_RED, prev_colors == COLOR_BLUE))
      b_to_r = np.count_nonzero(np.logical_and(
        g.colors == COLOR_BLUE, prev_colors == COLOR_RED))
      return (CYCLE_TWO, ) + g.count(prev_colors) + g.count() + (day, count_iso_r, count_iso_b, r_to_b, b_to_r)
    prev_two_colors = prev_colors
    prev_colors = g.colors

  r_to_b = np.count_nonzero(np.logical_and(
    g.colors == COLOR_RED, prev_two_colors == COLOR_BLUE))
  b_to_r = np.count_nonzero(np.logical_and(
    g.colors == COLOR_BLUE, prev_two_colors == COLOR_RED))

  return (INCONCLUSIVE, ) + g.count(prev_two_colors) + g.count() + (MAX_DAYS, count_iso_r, count_iso_b, r_to_b, b_to_r)


def main(n=10000, d=1, delta=0, n_trials=1000, save=True):
  """
  Runs multiple trials, where each trial has 3 steps:
  - generate a random graph G from the Erdos-Renyi G(n, d/n) distribution
  - run Majority Dynamics on G, with the n/2 + delta vertices initially assigned Red
  - collect relevant statistics about the trials, as described in the variable stats_list
  :param n: number of vertices
  :param d: expected degree (the graph is generated from Erdos-Renyi G(n, d/n))
  :param delta: advantage for Red
  :param n_trials: number of trials to run
  :param save: if True, the statistics in stats_list are saved for each trial and combined into a .npy file
  :return: None, but a summary is printed (even when save = False) that contains the statistics for each status:
      count: number of times that status occurs
      percentage: frequency that status occurs
      small(t-1): average of min(r(t-1), b(t-1)) over all trials
      big(t-1): average of max(r(t-1), b(t-1)) over all trials
      small(t): average of min(r(t), b(t)) over all trials
      big(t): average of max(r(t), b(t)) over all trials
      t, iso_r, iso_b, r->b, b->r: average of the corresponding fields described in stats_list
  """
  p = d/n
  print('Proceeding to run n_trials = {} trials on random graph G(n, p) '
        'for n = {}, p = {}/n.'.format(n_trials, n, d))
  print('Max days = {}. Advantage c = {}. Starting with {} Reds and {} Blues.'.format(
      MAX_DAYS, delta, math.ceil(n / 2 + delta), math.floor(n / 2 - delta)))
  print()

  # run n_trials trials
  records = np.empty((n_trials, len(info_to_index)))
  summary = {CYCLE_ONE: 0, CYCLE_TWO: 0, INCONCLUSIVE: 0}
  start = time.time()
  for i in range(1, n_trials + 1):
    result = trial(n, p, delta)
    records[i-1] = result
    summary[result[info_to_index['status']]] += 1
    print_progress_bar(start, i, n_trials, summary, CYCLE_ONE, CYCLE_TWO, INCONCLUSIVE)
  print()
  print('============== SUMMARY ==================')
  warnings.filterwarnings('ignore')
  # component_infos = np.average(records, axis=0)
  infos = dict()
  headline = ['status', 'count', 'percentage', 'small(t-1)', 'big(t-1)',
              'small(t)', 'big(t)', 't', 'iso_r', 'iso_b', 'r->b', 'b->r']
  for status in status_to_text.keys():
    records_with_stt = records[records[:, info_to_index['status']] == status]
    stt_count = records_with_stt.shape[0]
    infos[status] = {'status': status_to_text[status], 'count': stt_count,
                     'percentage': round(stt_count / records.shape[0], 2)}
    infos[status]['small(t-1)'] = round(np.average(
      np.minimum(records_with_stt[:, info_to_index['r(t-1)']], records_with_stt[:, info_to_index['b(t-1)']])
    ), 2)
    infos[status]['big(t-1)'] = round(np.average(
      np.maximum(records_with_stt[:, info_to_index['r(t-1)']], records_with_stt[:, info_to_index['b(t-1)']])
    ), 2)
    infos[status]['small(t)'] = round(np.average(
      np.minimum(records_with_stt[:, info_to_index['r(t)']], records_with_stt[:, info_to_index['b(t)']])
    ), 2)
    infos[status]['big(t)'] = round(np.average(
      np.maximum(records_with_stt[:, info_to_index['r(t)']], records_with_stt[:, info_to_index['b(t)']])
    ), 2)
    for field in headline[7:]:
      infos[status][field] = round(np.average(records_with_stt[:, info_to_index[field]]), 2)
  # Get proper width for each column
  col_width = {}
  for field in headline:
    col_width[field] = max(
      len(field), *[len(str(infos[status][field])) for status in status_to_text.keys()]
    )
  # Print headline
  headline_str = ''
  for field in headline:
    headline_str += '{0:>{1}}   '.format(field, col_width[field])
  print(headline_str)
  # Print info rows
  for status in status_to_text.keys():
    row_str = ''
    for field in headline:
      row_str += '{0:>{1}}   '.format(infos[status][field], col_width[field])
    print(row_str)

  print('-------------- END RESULTS -------------------------------------------------')
  print()

  # save records
  if save:
    import util
    data_subdir = '{}/{}_{}_{}'.format(DATA_DIR, n, util.d_encode(d), delta)
    os.makedirs(data_subdir, exist_ok=True)
    batch_list = [int(fname[- MAX_DIGITS - 4:-4]) for fname in glob.glob(data_subdir + '/*.npy')]
    first_batch = max(batch_list) + 1 if batch_list else 0
    np.save('{}/{:0{}}.npy'.format(data_subdir, first_batch, MAX_DIGITS), records)


if __name__ == '__main__':
  # A sample run for demonstration
  main(n=1000, d=9.5, delta=0, n_trials=10, save=False)


