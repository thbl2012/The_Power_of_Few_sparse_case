import math
import numpy as np
from random_graph import Graph


INDICATOR_EDGE = 2


def get_next(i, lower, upper, step=1):
  return (i - lower + step) % (upper - lower) + lower


# Divide m into k chunks of sizes in [a, a+1]
def even_divide(k, m):
  a = np.ones(k, dtype=int) * (m // k)
  a[np.random.choice(k, size=m % k, replace=False)] += 1
  return a


def get_random_level_sizes(n, min_center=0, max_center=3):
  n_odd = n % 2
  n_center = np.random.randint((min_center - n_odd + 1) // 2, (max_center - n_odd) // 2 + 1) * 2 + n_odd
  n_inner = (n - n_center) // 2
  n_outer = (n - n_center) // 2
  return n_center, n_inner, n_outer


class RandomCircleGraph(Graph):
  def __init__(self, size):
    super().__init__(size)
    self.levels = None
    self.circle_sizes = None
    self.circle_ends = None

  def generate(self, max_center=3, p_center=0.5, p_inner=0.5,
               p_outer=0.5, p_inout_st=0.5, p_inout_diag=0.5, p_cenin=0.5):
    n = self.size
    level_sizes = get_random_level_sizes(n, max_center=max_center)
    self.circle_sizes = level_sizes
    level_ends = np.concatenate(([0], np.cumsum(level_sizes)))
    self.circle_ends = level_ends
    # Generate edges among each level
    # Also assign level to each node
    self.edges = np.zeros((n, n), dtype=int)
    self.levels = np.zeros(n, dtype=int)
    probs = [p_center, p_inner, p_outer]
    for level in range(level_ends.shape[0] - 1):
      lower, upper = level_ends[level], level_ends[level + 1]
      for i in range(lower, upper):
        self.levels[i] = level
        i_next = get_next(i, lower, upper)
        self.set_edge(i, i_next, np.random.uniform(0, 1) < probs[level])
    # Randomly generate edges between inner and outer
    lvl_inner = 1
    n_inout = level_sizes[lvl_inner]
    lower, upper = level_ends[lvl_inner], level_ends[lvl_inner + 1]
    for i_in in range(lower, upper):
      i_out = i_in + n_inout
      i_in_next = get_next(i_in, lower, upper)
      i_out_next = i_in_next + n_inout
      self.set_edge(i_in, i_out, np.random.uniform(0, 1) < p_inout_st)
      diag_stt = np.random.choice(3, p=[1 - p_inout_diag, p_inout_diag / 2, p_inout_diag / 2])
      self.set_edge(i_in, i_out_next, diag_stt // 2)
      self.set_edge(i_in_next, i_out, diag_stt % 2)
    # Randomly generate edges between center and inner
    lvl_center = 0
    n_center = level_sizes[lvl_center]
    region_size = even_divide(n_center, n_inout)
    region_ends = np.concatenate(([0], np.cumsum(region_size)))
    for i in range(n_center):
      # lower = region_ends[i]
      # upper = region_ends[i + 1]
      for j in range(region_ends[i], region_ends[i + 1]):
        # print('Cenin: {} and {} mod {} = {}'.format(i, j, n_inout, (j - region_size[0] // 2) % n_inout))
        self.set_edge(i, (j - region_size[0] // 2) % n_inout + n_center,
                      np.random.uniform(0, 1) < p_cenin)
    self.edges *= 2
    np.fill_diagonal(self.edges, 1)


def arrange_circles(circle_sizes, circle_radii, position=None):
  if position is None:
    position = (0, 0)
  if circle_sizes[0] == 1:
    circle_radii[0] = 0
  cen_x, cen_y = position
  n = np.sum(circle_sizes)
  nodes_x = np.zeros(n)
  nodes_y = np.zeros(n)
  circle_ends = np.concatenate(([0], np.cumsum(circle_sizes)))
  for circ in range(circle_ends.shape[0] - 1):
    lower = circle_ends[circ]
    upper = circle_ends[circ + 1]
    angles = np.linspace(0, 2*np.pi, num=circle_sizes[circ], endpoint=False)
    nodes_x[lower:upper] = circle_radii[circ] * np.cos(angles) + cen_x
    nodes_y[lower:upper] = circle_radii[circ] * np.sin(angles) + cen_y
  return nodes_x, nodes_y


def arrange_circ_graph_edges(edges, nodes_x, nodes_y):
  n = nodes_x.shape[0]
  edge_count = np.count_nonzero(edges == INDICATOR_EDGE) // 2
  # print(edge_count)
  edges_x = np.zeros((edge_count, 2))
  edges_y = np.zeros((edge_count, 2))
  count = 0
  for i in range(n):
    lower_nodes = edges[i, :i]
    lower_neigh = (lower_nodes == INDICATOR_EDGE).nonzero()[0]
    for j in lower_neigh:
      # print(count, i, j)
      edges_x[count] = nodes_x[i], nodes_x[j]
      edges_y[count] = nodes_y[i], nodes_y[j]
      count += 1
  return edge_count, edges_x, edges_y



