
from collections import defaultdict
import numpy as np

def find_perms_naive(
  n_uses = 4,
  n_inner_folds = 20,
  n_test_split = 4,
  n_val_split = 2,
  seed = None
  ):

  if seed is not None:
    np.random.seed( seed )

  n_train_split = n_inner_folds - n_test_split

  train_splits = []
  test_splits = []
  val_splits = []

  count_of_uses = defaultdict( lambda : 0 )
  used_test_sets = []
  #lists_of_testsets = defaultdict( lambda : [] )
  for heldout_idx in range(n_inner_folds):

    options_left =  [j for j in range(n_inner_folds) if j != heldout_idx]
  
    #for prop_idx in range(0, n_uses):
    prop_idx = 0
    while prop_idx < n_uses:
      proposal = np.random.permutation( options_left ).tolist()

      test_split = proposal[:(n_test_split - 1)] + [ heldout_idx ]
      train_split = proposal[(n_test_split - 1):]
      val_split = train_split[:n_val_split]
      train_split = train_split[n_val_split:]

      #TODO: this is slow and lazy
      if set(test_split) in used_test_sets:
        continue
      else:
        used_test_sets.append(set(test_split))
        prop_idx += 1

      train_splits.append( train_split )
      test_splits.append( test_split )
      val_splits.append( val_split )

      for test_idx in test_split:
        count_of_uses[ test_idx ] += 1

  return list(zip(train_splits, val_splits, test_splits)), count_of_uses

def find_perms(
  n_uses = 4,
  n_inner_folds = 20,
  n_test_split = 4,
  n_val_split = 2,
  seed = 1919
  ):

  if seed is not None:
    np.random.seed( seed )

  n_train_split = n_inner_folds - n_test_split

  train_splits = []
  test_splits = []
  val_splits = []

  count_of_uses = defaultdict( lambda : 0 )
  used_test_sets = []
  #lists_of_testsets = defaultdict( lambda : [] )
  for heldout_idx in range(n_inner_folds):

    options_left =  [j for j in range(n_inner_folds) if j != heldout_idx]
  
    #for prop_idx in range(0, n_uses):
    prop_idx = count_of_uses[ heldout_idx ]
    while prop_idx < n_uses:
      proposal = np.random.permutation( options_left ).tolist()

      test_split = proposal[:(n_test_split - 1)] + [ heldout_idx ]
      train_split = proposal[(n_test_split - 1):]
      val_split = train_split[:n_val_split]
      train_split = train_split[n_val_split:]

      #TODO: this is slow and lazy
      if set(test_split) in used_test_sets:
        continue
      else:
        used_test_sets.append(set(test_split))
        prop_idx += 1

      train_splits.append( train_split )
      test_splits.append( test_split )
      val_splits.append( val_split )

      for test_idx in test_split:
        count_of_uses[ test_idx ] += 1

  return list(zip(train_splits, val_splits, test_splits)), count_of_uses



