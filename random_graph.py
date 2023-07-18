import numpy as np
import math, sys, time
from scipy.spatial.distance import squareform

MAX_DAYS = 6
SAVE_PERIOD = 1000


def color_count(coloring):
    size = coloring.shape[0]
    difference = coloring.sum()
    no_of_reds = (size +  difference) // 2
    no_of_blues = (size - difference) // 2
    return no_of_reds, no_of_blues


class Graph:
  def __init__(self, size):
    self.size = size
    self.colors = np.empty(size, dtype=np.int)
    self.true_edges = None
    self.edges = None

  def set_edge(self, i, j, e):
    self.edges[i, j] = e
    self.edges[j, i] = e

  # Sets first n/2 + c vertices to Red (1) and the rest to Blue (-1)
  def set_color(self, delta, coloring=None):
    pivot = math.ceil(self.size / 2 + delta)
    if coloring is None:
      self.colors[:pivot] = 1
      self.colors[pivot:] = -1
    elif coloring == 'max_deg':
      # assume edges are generated
      degrees = np.sum(self.edges, axis=0)
      degree_ranks = np.argsort(degrees)
      self.colors[:] = -1
      self.colors[degree_ranks >= self.size - pivot] = 1

  # Each vertex changes color according to the majority of their friends
  # Keeps color in case of tie (see comment of the function above)
  def transition(self):
    tmp = np.copy(self.colors)
    self.colors = (np.sum(self.edges * tmp, axis=1) > 0) * 2 - 1

  # Returns number of Red and Blue vertices respectively
  def count(self, coloring=None):
    if coloring is None:
      coloring = self.colors
    difference = coloring.sum()
    no_of_reds = (len(coloring) + difference) // 2
    no_of_blues = (len(coloring) - difference) // 2
    return no_of_reds, no_of_blues


class RandomGraph(Graph):
  # Generate edges randomly with probability p
  # Stores (2 * adjacency matrix) + I.
  # The I is for tie-breaking in transitioning
  def generate(self, p=.5):
    self.true_edges = squareform(np.random.random(size=self.size * (self.size - 1) // 2) < p)
    self.edges = self.true_edges * 2
    np.fill_diagonal(self.edges, 1)


# Simulate the election process on G(n, p) with advantage c
# Returns the winner and end day if ends within 20 days
# Otherwise return "Inconclusive"
def trial(n, p, c):
  g = RandomGraph(n)
  g.set_color(c)
  g.generate(p)
  for day in range(1, MAX_DAYS + 1):
    g.transition()
    n_reds, n_blues = g.count()
    if n_blues == 0:
      return 'Red', day
    if n_reds == 0:
      return 'Blue', day
  return 'Inconclusive', MAX_DAYS


# Repeat the election process for n_trials times and print aggregate results
def main(n, p, c, n_trials):
  print('Proceeding to run n_trials = {} trials on random graph G({}, {}).'.format(n_trials, n, p))
  print('Advantage c = {}. Starting with {} Reds and {} Blues.'.format(
      c, math.ceil(n / 2 + c), math.floor(n / 2 - c)))
  print()

  # run n_trials trials
  records = {}
  summary = {'Red': 0, 'Blue': 0, 'Inconclusive': 0}
  bar_length = 20
  start = time.time()
  for i in range(1, n_trials + 1):
    result = trial(n, p, c)
    if result not in records:
      records[result] = 1
    else:
      records[result] += 1
    summary[result[0]] += 1
    # print progress bar
    sys.stdout.write(
      '\rCompleted: {}/{} trials ({}%). R: {}, B: {}, I: {}. Elapsed: {} secs'.format(
        i, n_trials, math.floor((i / n_trials) * 100),
        summary['Red'], summary['Blue'], summary['Inconclusive'],
        round(time.time() - start, 2)))
    sys.stdout.flush()

  print()

  # print records
  print('==================== RESULTS =====================')
  print('Winner           Last day     Count      Frequency')
  print('---------------------------------------------')
  for result, count in sorted(records.items()):
    winner, day = result
    frequency = count / n_trials
    # summary[winner] += count
    print(
      '{:<12}      {:<4}         {:<6}      {:.2f}'.format(
        winner, day, count, frequency
      )
    )
  print()
  print('============== SUMMARY ==================')
  print('Winner           Count     Frequency')
  for winner, count in summary.items():
    print('{:<12}      {:<6}      {:.2f}'.format(
      winner, count, count / n_trials
    ))


# Read input from command line
def get_input(n, p, c, T):
  args = sys.argv[1:]
  names = ['n', 'p', 'c', 'n_trials']
  vals = [n, p, c, T]
  for i in range(4):
    if len(args) > i:
      vals[i] = type(vals[i])(args[i])
      print(
        'Input from command line: {} = {}'.format(names[i], vals[i])
      )
    else:
      print(
        'No input for {} provided. Using default {} = {}'.format(
          names[i], names[i], vals[i]
        )
      )
  return vals


if __name__ == '__main__':
  main(*get_input(10000, 0.5, 1, 100))
  # import time
  # start = time.time()
  # main(10000, 0.5, 1, 100)
  # end = time.time()
  # print('Running time: {} seconds'.format(start - end))
