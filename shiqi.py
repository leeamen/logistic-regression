#!/usr/bin/python
#coding:utf-8

from mylogistic import *
import numpy as np
import sys
import argparse

def MultiClassification(train_file_x, train_file_y, test_file_x, test_file_y):

  train_data = np.loadtxt(train_file_x, delimiter = ',', dtype = np.float)
  train_label_data = np.loadtxt(train_file_y, delimiter = ',', dtype = np.float)
  test_data = np.loadtxt(test_file_x, delimiter = ',', dtype = np.float)
  test_label_data = np.loadtxt(test_file_y, delimiter = ',', dtype = np.float)
#  train_data = np.column_stack((train_data, label_data))
#  np.random.shuffle(train_data)

  train_x = train_data
  train_y = train_label_data
  test_x = test_data
  test_y = test_label_data
  train_x = Normalizition(train_x)
  test_x = Normalizition(test_x)

  #number of iter
  param = {}
  #二分类
  iteration = 70
  param['objective'] = 'multi'
  param['learning_rate'] = 15
  param['num_iters'] = iteration
  param['num_class'] = 10
  param['lam'] = 1

  model = MyLRModel(param)
  model.Train(train_x, train_y)

  #预测(训练数据)
  pre_y = model.Predict(train_x)
  print '迭代次数:%d' %(iteration)
  print '训练集准确度为:%f' %(float(np.sum(train_y == pre_y)) / len(train_y))
  pred_testy = model.Predict(test_x)
#  print test_y
#  print pred_testy
  print '测试集准确率为:%f' %(float(np.sum(test_y == pred_testy))/len(test_y))

#  for i in range(0, len(train_x)):
#    j = np.random.permutation(len(train_x))[0]
#    pred = model.Predict(train_x[j,:].reshape(1,train_x.shape[1]))
#    print '该图像算法预测的数字是:', pred
#    ShowPicture(train_x, j)

if __name__ == '__main__':
  #参数读取
#  parser = argparse.ArgumentParser(description = '逻辑回归')
#  parser.add_argument('--train', help='训练文件')
#  parser.add_argument('--label', help='标签文件(可选)')
#
#  args = parser.parse_args()

#  if args.train == None:
#    parser.print_help()
#    exit()
  train_file_x, train_file_y, test_file_x, test_file_y = sys.argv[1:5]

  MultiClassification(train_file_x, train_file_y, test_file_x, test_file_y)

