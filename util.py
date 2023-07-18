import numpy as np
import math
import os


def d_encode(d, round_integers=True):
  if d == math.floor(d) and round_integers:
    return str(math.floor(d))
  else:
    return str(d).replace('.', 'd')


def load_data(n, d, delta, data_dir, file_index=-1, round_integers=True):
  import os
  target_dir = '{}/{}_{}_{}'.format(data_dir, n, d_encode(d, round_integers=round_integers), delta)
  if file_index == -1:
    return np.concatenate(
      [np.load('{}/{}'.format(target_dir, filename))
       for filename in os.listdir(target_dir) if filename[-4:] == '.npy'],
      axis=0
    )
  else:
    filename = os.listdir(target_dir)[file_index]
    return np.load('{}/{}'.format(target_dir, filename))
