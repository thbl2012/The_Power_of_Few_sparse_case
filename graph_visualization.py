import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

from random_graph import color_count
from RandomTree import *
from RandomCircleGraph import *

COLOR_RED = 1
COLOR_BLUE = -1
COLOR_CODES = {COLOR_RED: 'red', COLOR_BLUE: 'blue'}


def draw_graph(count, nodes_x, nodes_y, edges_x, edges_y, colors,
               edge_width=2, name_ver_pos=0.5, name_hor_pos=0.5,
               node_radius=5, fontsize=20):
  n = len(nodes_x)
  # Plot nodes using stored info
  scatters = plt.scatter(nodes_x, nodes_y, marker='o', s=node_radius ** 2, zorder=2,
                         color=[COLOR_CODES[colors[i]] for i in range(n)])
  for i in range(n):
    plt.annotate(str(i), (nodes_x[i], nodes_y[i]),
                 (nodes_x[i] + name_hor_pos, nodes_y[i] + name_ver_pos),
                 ha='center', va='center', weight='heavy',
                 fontsize=fontsize, color='white')
  # Plot edges using stored info
  for i in range(count):
    plt.plot(edges_x[i], edges_y[i], linewidth=edge_width, color='k', zorder=1)
  print('Day 0. Red: {}, Blue: {}'.format(*color_count(colors)))
  return scatters


class Index:
  def __init__(self, scatter, colors, ind=0, lower=0, upper=10):
    self.ind = ind
    self.scatter = scatter
    self.colors = colors
    self.lower = lower
    self.upper = upper

  def next(self, event):
    if self.ind < self.upper - 1:
      self.ind += 1
      self.scatter.set_color([COLOR_CODES[c] for c in self.colors[self.ind]])
      plt.draw()
    print('Day {}. Red: {}, Blue: {}'.format(self.ind, *color_count(self.colors[self.ind])))

  def prev(self, event):
    if self.ind > self.lower:
      self.ind -= 1
      self.scatter.set_color([COLOR_CODES[c] for c in self.colors[self.ind]])
      plt.draw()
    print('Day {}. Red: {}, Blue: {}'.format(self.ind, *color_count(self.colors[self.ind])))


def check_tree(n=10, root=0, delta=0):
  g = RandomTree(size=n)
  g.generate()
  g.set_color(delta=delta)
  edges = np.floor(g.edges / 2).astype(int)
  widths = np.zeros(n, dtype=int)
  get_width_at_root(edges, root, widths)
  count, nodes_x, nodes_y, edges_x, edges_y = arrange_tree(
    edges, widths, root, 0, edge_length=10)
  draw_graph(count, nodes_x, nodes_y, edges_x, edges_y, g.colors,
             edge_width=2, name_ver_pos=0, name_hor_pos=0,
             node_radius=20, fontsize=12)
  plt.show()


def check_circ_graph(n=10, delta=0):
  g = RandomCircleGraph(size=n)
  g.generate(max_center=3, p_center=1, p_inner=.5, p_outer=.5,
             p_inout_st=.5, p_inout_diag=.5, p_cenin=.5)
  g.set_color(delta=delta)
  nodes_x, nodes_y = arrange_circles(g.circle_sizes, [1, 2, 3])
  count, edges_x, edges_y = arrange_circ_graph_edges(g.edges, nodes_x, nodes_y)
  draw_graph(count, nodes_x, nodes_y, edges_x, edges_y, g.colors,
             edge_width=2, name_ver_pos=0, name_hor_pos=0,
             node_radius=20, fontsize=12)
  plt.show()


def draw_graph_transition(scatter, colors, n_days):
  callback = Index(scatter, colors, ind=0, lower=0, upper=n_days)
  ax_prev = plt.axes([0.7, 0.01, 0.1, 0.075])
  ax_next = plt.axes([0.81, 0.01, 0.1, 0.075])
  b_next = Button(ax_next, 'Next')
  b_next.on_clicked(callback.next)
  ax_next._button = b_next
  b_prev = Button(ax_prev, 'Previous')
  b_prev.on_clicked(callback.prev)
  ax_prev._button = b_prev


def check_tree_transition(n=10, root=0, delta=0, n_days=5):
  g = RandomTree(size=n)
  g.generate()
  g.set_color(delta=delta)
  edges = np.floor(g.edges / 2).astype(int)
  widths = np.zeros(n, dtype=int)
  get_width_at_root(edges, root, widths)
  colors = np.empty((n_days, n), dtype=int)
  colors[0] = g.colors
  for i in range(1, n_days):
    g.transition()
    colors[i] = g.colors
  count, nodes_x, nodes_y, edges_x, edges_y = arrange_tree(
    edges, widths, root, 0, edge_length=10)
  sc = draw_graph(count, nodes_x, nodes_y, edges_x, edges_y, colors[0],
                  edge_width=2, name_ver_pos=0, name_hor_pos=0,
                  node_radius=20, fontsize=12)
  draw_graph_transition(sc, colors, n_days)
  plt.show()


def check_circ_graph_transition(n=10, delta=0, n_days=5):
  g = RandomCircleGraph(size=n)
  g.generate(max_center=3, p_center=.8, p_inner=.8, p_outer=.8,
             p_inout_st=.8, p_inout_diag=.8, p_cenin=.8)
  g.set_color(delta=delta)
  colors = np.empty((n_days, n), dtype=int)
  colors[0] = g.colors.copy()
  for i in range(1, n_days):
    g.transition()
    colors[i] = g.colors.copy()
  nodes_x, nodes_y = arrange_circles(g.circle_sizes, [1, 2, 3])
  count, edges_x, edges_y = arrange_circ_graph_edges(g.edges, nodes_x, nodes_y)
  sc = draw_graph(count, nodes_x, nodes_y, edges_x, edges_y, colors[0],
                  edge_width=2, name_ver_pos=0, name_hor_pos=0,
                  node_radius=20, fontsize=12)
  draw_graph_transition(sc, colors, n_days)
  plt.show()


if __name__ == '__main__':
  # check_circ_graph_transition(n=25, n_days=10)
  check_tree_transition(n=15, n_days=10)
