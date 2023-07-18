import numpy as np
import math, sys, time, os, glob
from datetime import timedelta
# from collections import Counter
import warnings
from scipy.sparse.csgraph import connected_components
from random_graph import RandomGraph

MAX_DAYS = 100
COLOR_RED = 1
COLOR_BLUE = -1
CYCLE_ONE = 1
CYCLE_TWO = 2
INCONCLUSIVE = 0
status_to_text = {
  CYCLE_ONE: 'Cycle one', CYCLE_TWO: 'Cycle_two', INCONCLUSIVE: 'Inconclusive'
}

# info_to_index = {
#   'V_big': 0, 'E_big': 1, 'r(0)': 2, 'b(0)': 3, 'stt': 4,
#   'r(t-1)': 5, 'b(t-1)': 6, 'r(t)': 7, 'b(t)': 8,
#   '  t': 9, 'r->b': 10, 'b->r': 11, 'r(t-1)_deg': 12,
#   'b(t-1)_deg': 13, 'r(t)_deg': 14, 'b(t)_deg': 15,
#   'iso': 16, 'iso_b': 17, 'iso_rr': 18, 'iso_rb': 19, 'iso_bb': 20,
#   'stt_re': 21, 'r(t-1)_re': 22, 'b(t-1)_re': 23, 'r(t)_re': 24, 'b(t)_re': 25, 't_re': 26
# }

stats_list = [
  'V_big',  # number of vertices in the giant component
  'E_big',  # number of edges in the giant component
  'r(0)',  # number of initial Reds in the giant component
  'b(0)',  # number of initial Blues in the giant component
  'stt',  # status in the giant component:
          # 0: inconclusive after MAX_DAYS days, 1: period of 1, 2: period of 2
  'r(t-1)',  # number of Reds on the day before the first repeated day in the giant component if status = 2
  'b(t-1)',  # number of Blues on the day before the first repeated day in the giant component if status = 2
  'r(t)',  # number of Reds on the first repeated day in the giant component
  'b(t)',  # number of Blues on the first repeated day in the giant component
  '  t',  # the first repeated day in the giant component
  'r->b',  # number of Reds on day t that turn Blue on day t+1 in the giant component
  'b->r',  # number of Blues on day t that turn Red on day t+1 in the giant component
  'r(t-1)_deg',  # average degree of Reds on the day before the first repeated day in the giant component if status = 2
  'b(t-1)_deg',  # average degree of Blues on the day before the first repeated day in the giant component if status = 2
  'r(t)_deg',  # average degree of Reds on the first repeated day in the giant component
  'b(t)_deg',  # average degree Blues on the first repeated day in the giant component
  'iso',  # number of isolated vertices (not affected throughout the majority dynamics)
  'iso_b', # number of isolated Blues (not affected throughout the majority dynamics)
  'iso_rr',  # number of isolated Red-Red edges (not affected throughout the majority dynamics)
  'iso_rb',  # number of isolated Red-Blue edges (not affected throughout the majority dynamics)
  'iso_bb',  # number of isolated Blue-Blue edges (not affected throughout the majority dynamics)
  'stt_re',  # status in the complement of the giant component
  'r(t-1)_re',  # number of Reds on the day before the first repeated day in the complement of the giant component if status = 2
  'b(t-1)_re',  # number of Blues on the day before the first repeated day in the complement of the giant component if status = 2
  'r(t) _re',  # number of Reds on the first repeated day in the complement of the giant component
  'b(t)_re',  # number of Blues on the first repeated day in the complement of the giant component
  't_re'  # the first repeated day in the complement of the giant component
]
info_to_index = dict()
for i, field in enumerate(stats_list):
  info_to_index[field] = i

DATA_DIR = 'colors_on_components'
MAX_DIGITS = 6


# Info to store about each component:
# Size, No of edges, n_red, n_blue, n_red_end, n_blue_end, n_red_prev_end, n_blue_prev_end
# r_to_b, b_to_r, avg_r_deg_end, avg_b_deg_end,
def trial(n, d, delta):
  g = RandomGraph(n)
  g.set_color(delta)
  g.generate(d / n)
  results = np.empty(len(info_to_index))

  # find connected components
  n_components, component = connected_components(
    np.maximum(g.edges - 1, 0), directed=False, return_labels=True)
  component_matrix = np.zeros((n_components, n), dtype=bool)
  for i in range(n):
    component_matrix[component[i], i] = True

  # identify largest component
  component_sizes = np.sum(component_matrix, axis=1)
  k_max = np.argmax(component_sizes)
  g_max = RandomGraph(component_sizes[k_max])
  in_g_max = component_matrix[k_max]
  g_max.edges = g.edges[in_g_max][:, in_g_max]
  g_max.colors = g.colors[in_g_max]

  # run the process on the largest component
  pre_info = (component_sizes[k_max], np.sum(np.maximum(g_max.edges - 1, 0)) // 2) + g_max.count()
  post_info = trial_on_component(g_max)
  results[:-11] = pre_info + post_info

  # identify isolated edges
  is_iso_edge = component_sizes == 2
  iso_edges = component_matrix[is_iso_edge]
  iso_edges_colors = np.dot(iso_edges, g.colors)
  results[info_to_index['iso_rr']] = np.count_nonzero(iso_edges_colors == 2)
  results[info_to_index['iso_rb']] = np.count_nonzero(iso_edges_colors == 0)
  results[info_to_index['iso_bb']] = np.count_nonzero(iso_edges_colors == -2)

  # identify isolated vertices
  is_iso_ver = component_sizes == 1
  iso_vers = component_matrix[is_iso_ver]
  iso_ver_colors = np.dot(iso_vers, g.colors)
  results[info_to_index['iso']] = iso_vers.shape[0]
  results[info_to_index['iso_b']] = np.count_nonzero(iso_ver_colors == -1)

  # identify the remaining subgraph
  n_rest = n - g_max.size - iso_edges.shape[0] - iso_vers.shape[0]
  g_rest = RandomGraph(n_rest)
  is_comp_in_g_rest = np.logical_not(np.logical_or(is_iso_edge, is_iso_ver))
  is_comp_in_g_rest[k_max] = False
  in_g_rest = is_comp_in_g_rest[component]
  g_rest.edges = g.edges[in_g_rest][:, in_g_rest]
  g_rest.colors = g.colors[in_g_rest]

  # run the process on the remaining component
  rest_info = trial_on_component(g_rest)[4:10]
  results[-6:] = rest_info

  return results


def trial_on_component(g_k):
  prev_colors = g_k.colors
  prev_two_colors = g_k.colors
  status = None
  # Number of nodes that are in cycle two at prev day
  r_to_b = 0
  b_to_r = 0
  day = 0
  for day in range(1, MAX_DAYS + 1):
    prev_colors = g_k.colors
    g_k.transition()
    is_cycle_1 = np.array_equal(g_k.colors, prev_colors)
    is_cycle_2 = day > 1 and np.array_equal(g_k.colors, prev_two_colors)
    if is_cycle_1:
      # Number of nodes that are in cycle two at prev day
      r_to_b = 0
      b_to_r = 0
      status = CYCLE_ONE
      break
    if is_cycle_2:
      # Number of nodes that are in cycle two at prev day
      r_to_b = np.count_nonzero(np.logical_and(
        g_k.colors == COLOR_RED, prev_colors == COLOR_BLUE))
      b_to_r = np.count_nonzero(np.logical_and(
        g_k.colors == COLOR_BLUE, prev_colors == COLOR_RED))
      status = CYCLE_TWO
      break
    prev_two_colors = prev_colors
  if status is None:
    status = INCONCLUSIVE
  # Common calculations
  r_prev, b_prev = g_k.count(prev_colors)
  r, b = g_k.count()
  # The mean has to be -1 then /2 because of the way edges are stored
  avg_r_deg = round(np.mean(np.sum(g_k.edges[g_k.colors == COLOR_RED], axis=1) - 1) / 2, 2)
  avg_b_deg = round(np.mean(np.sum(g_k.edges[g_k.colors == COLOR_BLUE], axis=1) - 1) / 2, 2)
  avg_r_prev_deg = round(np.mean(np.sum(g_k.edges[prev_colors == COLOR_RED], axis=1) - 1) / 2, 2)
  avg_b_prev_deg = round(np.mean(np.sum(g_k.edges[prev_colors == COLOR_BLUE], axis=1) - 1) / 2, 2)
  return (status, r_prev, b_prev, r, b, day, r_to_b, b_to_r,
          avg_r_prev_deg, avg_b_prev_deg, avg_r_deg, avg_b_deg)


def main_single(n=10000, d=1, delta=0):
  """
  Run a single trial and print out the information
  """
  warnings.filterwarnings('ignore')
  component_infos = trial(n, d, delta)
  # Get proper width for each column
  col_width = {}
  for info in info_to_index.keys():
    col_width[info] = max(len(info), len(str(component_infos[info_to_index[info]])))
  # Print headline string
  headline = ''
  for info in info_to_index.keys():
    headline += '{0:>{1}}   '.format(info, col_width[info])
  print(headline)
  # Print each result row
  result = ''
  for info in info_to_index.keys():
    content = component_infos[info_to_index[info]]
    if content.is_integer():
      content = int(content)
    # else:
    #   content = round(content, 2)
    result += '{0:>{1}}   '.format(content, col_width[info])
  print(result)


def print_progress_bar(start_time, n_trials_done, n_trials_total):
  # print progress bar
  sys.stdout.write(
    '\rCompleted: {}/{} trials. Elapsed: {}'.format(
      n_trials_done, n_trials_total,
      timedelta(seconds=time.time() - start_time)))
  sys.stdout.flush()


def main(n=10000, d=1, delta=0, n_trials=100, save=False):
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
  :return: None, but a summary is printed (even when save = False) that contains the statistics for each status.
  For each field in stats_list, the average for that field among all trials is printed, grouped by status
  """
  print('Proceeding to run n_trials = {} trials on random graph G(n, d/n) '
        'for n = {}, d = {}.'.format(n_trials, n, d))
  print('Max days = {}. Advantage c = {}. Starting with {} Reds and {} Blues.'.format(
    MAX_DAYS, delta, math.ceil(n / 2 + delta), math.floor(n / 2 - delta)))
  print()

  # run n_trials trials
  records = np.empty((n_trials, len(info_to_index)))  # dict format: (status, n_red, n_blue): number of occurences
  start = time.time()
  for i in range(1, n_trials + 1):
    records[i - 1] = trial(n, d, delta)
    print_progress_bar(start, i, n_trials)
  print()
  print('============== SUMMARY ==================')
  warnings.filterwarnings('ignore')
  component_infos = np.average(records, axis=0)
  # Get proper width for each column
  col_width = {}
  for info in info_to_index.keys():
    col_width[info] = max(len(info), len(str(component_infos[info_to_index[info]])))
  # Print headline string
  headline = ''
  for info in info_to_index.keys():
    headline += '{0:>{1}}   '.format(info, col_width[info])
  print(headline)
  # Print each result row
  result = ''
  for info in info_to_index.keys():
    content = component_infos[info_to_index[info]]
    if content.is_integer():
      content = int(content)
    # else:
    #   content = round(content, 2)  b
    result += '{0:>{1}}   '.format(content, col_width[info])
  print(result)
  print('-------------- END RESULTS -------------------------------------------------')
  print()

  # save records
  if save:
    data_subdir = '{}/{}_{}_{}'.format(DATA_DIR, n, str(d).replace('.', 'd'), delta)
    os.makedirs(data_subdir, exist_ok=True)
    print('Results saved in {}'.format(data_subdir))
    batch_list = [int(fname[- MAX_DIGITS - 4:-4]) for fname in glob.glob(data_subdir + '/*.npy')]
    first_batch = max(batch_list) + 1 if batch_list else 0
    np.save('{}/{:0{}}.npy'.format(data_subdir, first_batch, MAX_DIGITS), records)


if __name__ == '__main__':
  # A sample run for demonstration:
  main(10000, d=10, delta=2000, n_trials=10, save=False)
