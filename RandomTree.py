import math
import numpy as np
from random_graph import Graph

INDICATOR_PARENT = 2
INDICATOR_CHILD = 1


class RandomTree(Graph):
  def generate(self):
    n = self.size
    self.edges = np.zeros((n, n), dtype=np.int32)
    # Initial partition: (tree, remains) = ([0], all nodes > 0)
    curr_partition = np.arange(0, n)
    curr_tree_size = 1
    # Each step, choose a random node in tree and a random node in remains and connect them,
    # then add the connected node to tree by swapping it with the node at index [curr_tree_size]
    for curr_tree_size in range(1, n):
      i = np.random.choice(curr_partition[:curr_tree_size], size=1)
      index = np.random.randint(curr_tree_size, n)
      j = curr_partition[index]
      self.edges[i, j] = 2
      self.edges[j, i] = 2
      curr_partition[curr_tree_size], curr_partition[index] = curr_partition[index], curr_partition[curr_tree_size]
    np.fill_diagonal(self.edges, 1)


# Get the width (= number of leaves) at this rooted tree and each subtree and save in result
# Also update parent-child relation according to this rooting
def get_width_at_root(edges, root, result):
  children = (edges[root] == INDICATOR_CHILD).nonzero()[0]
  if len(children) == 0:  # If a leaf is reached
    return 0  # Leaf has width 0
  for child in children:
    # Update parent-child relation
    edges[child, root] = INDICATOR_PARENT
    # If child is a leaf, it reports 1 back to its parent instead
    result[root] += max(1, get_width_at_root(edges, child, result))
  return result[root]


def child_position_in_mid_arc(x_root, y_root, low, high, length=1):
  mid = (low + high) / 2
  return round(x_root + length * math.cos(mid), 3), round(y_root + length * math.sin(mid), 3)


# Returns list of nodes in the subtree from this root and list of edges to plot
def arrange_tree(edges, widths, root, edge_count, position=None,
                 low_angle=0, high_angle=2 * math.pi, edge_length=1,
                 nodes_x=None, nodes_y=None, edges_x=None, edges_y=None):
  if position is None:
    position = (0, 0)
  # Empty arrays to store result of arrange_tree
  n = len(widths)
  nodes_x = np.empty(n) if nodes_x is None else nodes_x
  nodes_y = np.empty(n) if nodes_y is None else nodes_y
  edges_x = np.empty((n, 2)) if edges_x is None else edges_x
  edges_y = np.empty((n, 2)) if edges_y is None else edges_y
  x_root, y_root = position
  nodes_x[root] = x_root
  nodes_y[root] = y_root
  children = (edges[root] == INDICATOR_CHILD).nonzero()[0]
  if len(children) != 0:
    arc = high_angle - low_angle
    low = low_angle
    for i, child in enumerate(children):
      high = low + arc * max(1, widths[child]) / widths[root]
      x_child, y_child = child_position_in_mid_arc(x_root, y_root, low, high, length=edge_length)
      edges_x[edge_count] = x_child, x_root
      edges_y[edge_count] = y_child, y_root
      edge_count, _, _, _, _ = arrange_tree(
        edges, widths, child, edge_count + 1, position=(x_child, y_child),
        low_angle=low, high_angle=high, edge_length=edge_length,
        nodes_x=nodes_x, nodes_y=nodes_y, edges_x=edges_x, edges_y=edges_y
      )
      low = high
  return edge_count, nodes_x, nodes_y, edges_x, edges_y