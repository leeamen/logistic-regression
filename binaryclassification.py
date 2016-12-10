#!/usr/bin/python
#coding:utf-8

from mylogistic import *
import numpy as np

#main function
def BinaryClassification(trainfile, labelfile):
  print '训练文件:%s' %(trainfile)
  
  train_data = np.loadtxt(trainfile, delimiter = ',', dtype = object)
  #print 'train_data',train_data
  
  train_x = np.array(train_data[:,[0,1]], dtype = np.float)
  train_y = np.array(train_data[:,2], dtype = np.int)
  #print 'train_x', train_x
  #print 'trian_y', train_y
  
  #number of iter
  param = {}
  #二分类
  param['objective'] = 'binary'
  param['learning_rate'] = 1
  param['num_iters'] = 100
  
  
  model = MyLRModel(param)
  model.Train(train_x, train_y)

  #预测(训练数据)
  pre_y = model.Predict(train_x)
  print pre_y.shape,train_y.shape
  print '准确度为:%f' %(float(np.sum(train_y == pre_y)) / len(train_y))
  
  #画图
  figure()
  #title('逻辑回归（正则化，二维转高维达到线性可分）')
  title('logistic regression.\n(tranform low dimensions to high dimensions)')
  xlabel('x')
  ylabel('y')
  PlotData(train_x, train_y)
  #画等高线
  PlotBoundary(model.theta)
  
  #第三个标签打不出来
  legend(['y = 1', 'y = 0', 'decision boundary']) 
  show()

def MultiClassification(trainfile, labelfile):
  #
  print '训练文件:%s' %(trainfile)
  print '标签文件:%s' %(labelfile)

  train_data = np.loadtxt(trainfile, delimiter = ',', dtype = np.float)
  label_data = np.loadtxt(labelfile, delimiter = ',', dtype = np.int)

  train_x = train_data
  train_y = label_data

  #显示图片随机抽取100张
  ShowPictures(train_x)

  #number of iter
  param = {}
  #二分类
  param['objective'] = 'multi'
  param['learning_rate'] = 1
  param['num_iters'] = 100
  param['num_class'] = 10
  param['lam'] = 1

  model = MyLRModel(param)
  model.Train(train_x, train_y)

  #预测(训练数据)
  pre_y = model.Predict(train_x)
 # print pre_y.shape,train_y.shape
 # print sum(pre_y)
  print '准确度为:%f' %(float(np.sum(train_y == pre_y)) / len(train_y))

  for i in range(0, len(train_x)):
    j = np.random.permutation(5000)[0]
    pred = model.Predict(train_x[j,:].reshape(1,train_x.shape[1]))
    print '该图像算法预测的数字是:', pred
    ShowPicture(train_x, j)

import sys
import argparse
if __name__ == '__main__':
  #参数读取
  parser = argparse.ArgumentParser(description = '逻辑回归')
  parser.add_argument('--train', help='训练文件')
  parser.add_argument('--label', help='标签文件(可选)')

  args = parser.parse_args()

  if args.train == None:
    parser.print_help()
    exit()

#  MultiClassification(args.train, args.label)
  BinaryClassification(args.train, args.label)

