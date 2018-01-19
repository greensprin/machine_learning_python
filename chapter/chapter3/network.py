# coding:utf-8

import numpy as np
import sys
#original import file
sys.path.append("/home/momiki/work/python/machine_learning_python/data")
import download_mnist

def sigmoid(x):
# シグモイド関数(0~1の間を取る)
  return 1 / (1 + exp(-x))

def relu(x):
# Relu関数.比較して大きい方を出力（0以下のものはすべて0）
  return np.maximum(0, x)

if __name__ == '__main__':
# data load
  trn_data, trn_lbl, tst_data, tst_lbl = download_mnist.all_data_load()
  print(trn_data.shape)
  print(trn_lbl)
  print(tst_data.shape)
  print(tst_lbl)
