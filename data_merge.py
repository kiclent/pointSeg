

import sys
sys.path.append('../')
import os
from lib.utils import file_merge

train_csv_path = '../inputs/dataset/training/'
test_csv_path = '../inputs/TestSet'

train_npy_path = '../inputs/npy/train'
test_npy_path = '../inputs/npy/test'

if not os.path.exists(train_npy_path):
    os.makedirs(train_npy_path)

if not os.path.exists(test_npy_path):
    os.makedirs(test_npy_path)

# file_merge(train_csv_path, train_npy_path, is_training=True)
file_merge(test_csv_path, test_npy_path, is_training=False)


