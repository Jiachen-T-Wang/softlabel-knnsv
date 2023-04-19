import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms

# general
import pandas as pd 
import numpy as np 
import copy
import pickle
import sys
import time
import os
import random
import pdb

from helper import *
from prepare_data import *
import config


# python applications_knn.py --task mislabel_detect --dataset creditcard --value_type KNN-SV-JW --n_data 2000 --n_val 2000 --flip_ratio 0.1 --K 5

# python applications_knn.py --task mislabel_detect --dataset Dog_vs_CatFeature --value_type KNN-SV-JW --n_data 2000 --n_val 200 --flip_ratio 0.1 --K 5


import argparse

parser = argparse.ArgumentParser('')

parser.add_argument('--dataset', type=str)
parser.add_argument('--value_type', type=str)
parser.add_argument('--n_data', type=int, default=2000)
parser.add_argument('--n_val', type=int, default=200)
parser.add_argument('--flip_ratio', type=float, default=0)
parser.add_argument('--task', type=str)
parser.add_argument('--K', type=int, default=5)

args = parser.parse_args()

dataset = args.dataset
value_type = args.value_type
n_data = args.n_data
n_val = args.n_val
flip_ratio = float(args.flip_ratio) * 1.0
task = args.task
K = args.K

big_dataset = config.big_dataset
OpenML_dataset = config.OpenML_dataset

save_dir = 'result/'

if task=='mislabel_detect':
  x_train, y_train, x_val, y_val = get_processed_data(dataset, n_data, n_val, flip_ratio)
else:
  exit(1)


data_lst = []


for i in range(1):

  start = time.time()

  if value_type == 'KNN-SV-RJ':
    sv = knn_shapley_RJ(x_train, y_train, x_val, y_val, K=K)
  elif value_type == 'KNN-SV-JW':
    sv = knn_shapley_JW(x_train, y_train, x_val, y_val, K=K)
  else:
    exit(1)

  print('Data Value Computed; Value Name: {}; Runtime: {} s'.format( value_type, np.round(time.time()-start, 3) ))

  if task in ['mislabel_detect']:
    acc1, acc2, auc = kmeans_f1score(sv, cluster=False), kmeans_f1score(sv, cluster=True), kmeans_aucroc(sv)
    data_lst.append( [acc1, acc2, auc] )


if task in ['mislabel_detect']:

  print('Task: {}'.format(task))
  
  data_lst = np.array(data_lst)

  f1_rank, f1_rank_std = np.round( np.mean(data_lst[:, 0]), 3), np.round( np.std(data_lst[:, 0]), 3)
  f1_cluster, f1_cluster_std = np.round( np.mean(data_lst[:, 1]), 3), np.round( np.std(data_lst[:, 1]), 3)
  auc, std_auc = np.round( np.mean(data_lst[:, 2]), 3), np.round( np.std(data_lst[:, 2]), 3)
  
  print('*** {} F1-Rank: {}, F1-Cluster: {}, AUROC: {} ***'.format(value_type, f1_rank, f1_cluster, auc))
