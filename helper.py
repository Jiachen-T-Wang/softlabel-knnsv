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
from os.path import exists
import warnings

from tqdm import tqdm

import scipy
from scipy.special import beta, comb
from random import randint

from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

import config


big_dataset = config.big_dataset
OpenML_dataset = config.OpenML_dataset

save_dir = 'result/'


def kmeans_f1score(value_array, cluster=False):

  n_data = len(value_array)

  if cluster:
    X = value_array.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    min_cluster = min(kmeans.cluster_centers_.reshape(-1))
    pred = np.zeros(n_data)
    pred[value_array < min_cluster] = 1
  else:
    threshold = np.sort(value_array)[int(0.1*n_data)]
    pred = np.zeros(n_data)
    pred[value_array < threshold] = 1
    
  true = np.zeros(n_data)
  true[:int(0.1*n_data)] = 1
  return f1_score(true, pred)



def kmeans_aucroc(value_array):

  n_data = len(value_array)    
  true = np.zeros(n_data)
  true[int(0.1*n_data):] = 1
  return roc_auc_score(true, value_array)


def kmeans_aupr(value_array, cluster=False):

  n_data = len(value_array)

  if cluster:
    X = value_array.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    min_cluster = min(kmeans.cluster_centers_.reshape(-1))
    pred = np.zeros(n_data)
    pred[value_array < min_cluster] = 1
  else:
    threshold = np.sort(value_array)[int(0.1*n_data)]
    pred = np.zeros(n_data)
    pred[value_array < threshold] = 1
    
  true = np.zeros(n_data)
  true[:int(0.1*n_data)] = 1
  return average_precision_score(true, pred)



def normalize(val):
  v_max, v_min = np.max(val), np.min(val)
  val = (val-v_min) / (v_max - v_min)
  return val


def rank_neighbor(x_test, x_train):
  distance = np.array([np.linalg.norm(x - x_test) for x in x_train])
  return np.argsort(distance)


# x_test, y_test are single data point
def knn_shapley_RJ_single(x_train_few, y_train_few, x_test, y_test, K):
  N = len(y_train_few)
  sv = np.zeros(N)
  rank = rank_neighbor(x_test, x_train_few)
  sv[int(rank[-1])] += int(y_test==y_train_few[int(rank[-1])]) / N

  for j in range(2, N+1):
    i = N+1-j
    sv[int(rank[-j])] = sv[int(rank[-(j-1)])] + ( (int(y_test==y_train_few[int(rank[-j])]) - int(y_test==y_train_few[int(rank[-(j-1)])])) / K ) * min(K, i) / i

  return sv


# Original KNN-Shapley proposed in http://www.vldb.org/pvldb/vol12/p1610-jia.pdf
def knn_shapley_RJ(x_train_few, y_train_few, x_val_few, y_val_few, K):
  
  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  for i in range(n_test):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += knn_shapley_RJ_single(x_train_few, y_train_few, x_test, y_test, K)

  return sv


# x_test, y_test are single data point
def knn_shapley_JW_single(x_train_few, y_train_few, x_test, y_test, K):
  N = len(y_train_few)
  sv = np.zeros(N)
  rank = rank_neighbor(x_test, x_train_few).astype(int)
  C = max(y_train_few)+1

  c_A = np.sum( y_test==y_train_few[rank[:N-1]] )

  const = np.sum([ 1/j for j in range(1, min(K, N)+1) ])

  sv[rank[-1]] = (int(y_test==y_train_few[rank[-1]]) - c_A/(N-1)) / N * ( np.sum([ 1/(j+1) for j in range(1, min(K, N)) ]) ) + (int(y_test==y_train_few[rank[-1]]) - 1/C) / N

  for j in range(2, N+1):
    i = N+1-j
    coef = (int(y_test==y_train_few[int(rank[-j])]) - int(y_test==y_train_few[int(rank[-(j-1)])])) / (N-1)

    sum_K3 = K

    sv[int(rank[-j])] = sv[int(rank[-(j-1)])] + coef * ( const + int( N >= K ) / K * ( min(i, K)*(N-1)/i - sum_K3 ) )

  return sv


# Soft-label KNN-Shapley proposed in https://arxiv.org/abs/2304.04258 
def knn_shapley_JW(x_train_few, y_train_few, x_val_few, y_val_few, K):
  
  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  for i in range(n_test):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += knn_shapley_JW_single(x_train_few, y_train_few, x_test, y_test, K)

  return sv



# x_test, y_test are single data point
def knn_shapley_JW_reg_single(x_train_few, y_train_few, x_test, y_test, K):
  N = len(y_train_few)
  sv = np.zeros(N)
  rank = rank_neighbor(x_test, x_train_few).astype(int)

  # Compute phi_N
  y_N = y_train_few[rank[-1]]
  y_sum = np.sum(y_train_few) - y_N
  y_square_sum = np.sum(y_train_few**2) - y_N**2

  star_sum = 0
  for j in range(1, K):
    term1 = (2*j + 1) / (j**2 * (j + 1)**2)
    term2 = (j * (j - 1) / ((N - 1) * (N - 2))) * y_sum**2
    term3 = (j * (N - j - 1) / ((N - 1) * (N - 2))) * y_square_sum
    term4 = (-2 * y_N / (j + 1)**2 - 2 * y_test / (j * (j + 1))) * (j / (N - 1)) * y_sum
    term5 = (y_N / (j + 1) - 2 * y_test) * (-y_N / (j + 1))
    star_sum += term1 * (term2 + term3) + term4 + term5

  sv[rank[-1]] = star_sum/N + (y_test**2 - (y_N-y_test)**2)/N


  for k in range(2, N+1):
    i = N+1-k

    # Compute A1
    A1_sum_part = np.sum([1 / j**2 for j in range(1, K + 1)])
    A1 = A1_sum_part + (1 / K**2) * ((N - 1) * min(K, i) / i - K)

    # Compute A2
    first_sum = np.sum([y_train_few[rank[l-1]] for l in range(1, N+1) if l!=i and l!=i+1])
    second_sum = np.sum([j / (j + 1)**2 for j in range(1, K)])
    first_term = (1 / (N - 2)) * first_sum * second_sum

    sum_up_to_i_minus_1 = np.sum([y_train_few[rank[l-1]] for l in range(1, i)])
    common_term = - ((K - 1) * K) / (2 * (N - 2))

    if i == 1:
      third_term = 0
    else:
      third_term = ((N - 1) * min(K, i) * min(K - 1, i - 1)) / (2 * (i - 1) * i)

    second_term = (
        (1 / K**2) * (
            sum_up_to_i_minus_1 * (third_term + common_term)
            + np.sum([y_train_few[rank[l-1]] * (((N-1)*min(K, l-1) * min(K-1, l-2)) / (2*(l-1)*(l-2)) + common_term) for l in range(i+2, N+1)])
        )
    )

    A2 = first_term + second_term

    # Compute A3
    A3_sum_part = np.sum([1 / j for j in range(1, K + 1)])
    A3 = A3_sum_part + min(K, i) * (N - 1) / (i * K) - 1

    yi, yip1 = y_train_few[rank[i-1]], y_train_few[rank[i]]

    sv[int(rank[-k])] = sv[int(rank[-(k-1)])] + (yip1-yi)/(N-1) * ( (yi+yip1)*A1 + 2*A2 - 2*y_test*A3 )


  return sv


def knn_shapley_JW_reg(x_train_few, y_train_few, x_val_few, y_val_few, K):

  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  for i in range(n_test):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += knn_shapley_JW_reg_single(x_train_few, y_train_few, x_test, y_test, K)

  return sv
