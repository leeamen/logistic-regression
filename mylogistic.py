#!/usr/bin/python
#coding:utf-8

import numpy as np
import copy
class MyLRModel:
  #param = {'objective':'binary', 'learning_rate':1, 'lam':1, 'num_iters':100}
  def __init__(self, param):
    if not param:
      print 'Error, 需要设置参数'
      exit()

    self.param = copy.deepcopy(param)

    #alpha 正则参数
    self.param['alpha'] = 1.0
    if not self.param.has_key('learning_rate'): self.param['learning_rate'] = 1
    if not self.param.has_key('num_iters'): self.param['num_iters'] = 100
    if not self.param.has_key('lam'): self.param['lam'] = 1
    if not self.param.has_key('objective'): print 'Error, 参数错误:objective';exit()

    #theta
    self.theta = None

    #设置参数
    if self.param['objective'] == 'multi':
      if self.param['num_class'] == None or self.param['num_class'] <=2:
        print 'Error, 参数错误:num_class',num_class
        exit()
    else:
      self.param['num_class'] = 1

  def Train(self, x, y):
    try:
      if x.shape[1] == 2:
        #多项式特征，达到高维线性可分的效果
        x = self.MapFeature(x[:,0], x[:,1])
        self.param['num_feature'] = x.shape[1]
      else:
        x = self.AddX0(x)
        self.param['num_feature'] = x.shape[1]
    except:
      x = self.AddX0(x)
      self.param['num_feature'] = 2

    #init theta
    self.theta = np.zeros((self.param['num_feature'], self.param['num_class']), dtype = np.float)

    if self.param['num_class'] == 1:
      self.GradientDescent(x, y , self.theta[:,0])
    else:
      #多分类
      for i in range(0, self.param['num_class']):
        self.GradientDescent(x, np.array(y == i, dtype = np.int), self.theta[:,i])

  def Predict(self, x):
    if x.shape[1] == 2:
      x = self.MapFeature(x[:,0], x[:,1])
    else:
      x = self.AddX0(x)

    #hypothesis
    h = self.hypothesis(x, self.theta)
    #print self.theta

    #用概率分类
    if self.param['num_class'] == 1:
      return np.array(h >= 0.5, dtype = np.int).T
    elif self.param['num_class'] > 1:
      result_class = np.zeros(len(x), dtype = np.int)
      for i in range(0, len(x)):
       # print h
       # print h[i]
       # print np.max(h[i])
        result_class[i] = np.int(np.where(h[i] == np.max(h[i]))[0])
      return result_class
    else:
      print 'Error, 分类个数错误:',self.param['num_class']
      return None

  def AddX0(self, x):
    real_x = np.ones(len(x), dtype = np.float)
    return np.column_stack((real_x, x))

  def GradientDescent(self, x, y, theta):
    cost_j = np.zeros(self.param['num_iters'], dtype = np.float)
    cost_j[0] = self.cost(x, y, theta)

    for i in range(1, self.param['num_iters']):
      #梯度下降
      grad = self.gradient(x, y, theta)
      theta1 = theta - self.param['learning_rate'] * grad
      cost_j[i] = self.cost(x, y, theta1)

      #判断是否到达最小值
#      if cost_j[i-1] - cost_j[i] <= 0.001:
#        break
      theta[:] = theta1[:]
      print 'iteration %d | cost:%f |loss:%f' %(i, cost_j[i], cost_j[i-1] - cost_j[i])

      #当前theta的梯度值
    print 'iteration over!', cost_j[i]
    #return theta

  #训练数据分类两个特征扩展多个高维特征
  @classmethod
  def MapFeature(self, x1, x2):
    degree = 6
  
    out = None
    try:
      out = np.ones(len(x1), dtype = np.float)
    except:
      out = np.ones(1, dtype = np.float)
  
    for i in range(1, degree+1):
      for j in range(0, i+1):
        col = (x1 ** j) * (x2 **(i-j))
        out = np.column_stack((out, col))
  
    return out
  
  def hypothesis(self, x, theta):
    #print np.dot(x, theta)
    return self.sigmoid(np.dot(x, theta))

  def sigmoid(self, x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))
  
  def cost(self, x, y, theta):
    #m records
    m = len(x)
  
    #
    h = self.hypothesis(x, theta)
    #cost function
    vector_j = y * np.log(h) + (1.0 - y) * np.log(1.0 - h)
  
    #regular item
    cost_j = -1.0 / m * np.sum(vector_j) + self.param['lam'] / (2.0 * m) * np.dot(theta[1:],theta[1:])

    return cost_j
  
  def gradient(self, x, y, theta):
    m = len(x)
  
    h = self.hypothesis(x, theta)
  
    grad = np.zeros(len(theta), dtype = np.float)
    #theta zero
    grad[0] = 1.0 / m * np.sum((h - y) * x[:,0])
  
    for i in range(1, len(theta)):
      grad[i] = 1.0 / m * np.sum((h-y) * x[:,i]) + self.param['lam'] / m * theta[i]

    return grad

def Normalizition(x):
  mu = np.nanmean(x, axis = 0)
  sigma = np.nanstd(x, axis = 0)
  x_norm = (x - mu) / sigma
  return x_norm

#画点
import numpy as np
from matplotlib.pyplot import *
from matplotlib.mlab import *
def PlotData(X,label):
  #画图

  x = X[:,0]
  y = X[:,1]
  
  pos = find(label == 1)
  neg = find(label == 0)
  
  plot(x[pos], y[pos], '+r', markersize = 5)
  plot(x[neg], y[neg], 'xg', markersize = 3)

def PlotBoundary(theta):
  #Here is the grid range
  u = np.linspace(-1, 1.5, 50)
  v = np.linspace(-1, 1.5, 50)

  z = np.zeros((len(u), len(v)));
  # Evaluate z = theta*x over the grid
  for i in range(0, len(u)):
      for j in range(0, len(v)):
        map_x = MyLRModel.MapFeature(u[i], v[j])
        z[i,j] = np.dot(map_x, theta)

  # important to transpose z before calling contour
  #z = z.T

  # Plot z = 0
  # Notice you need to specify the range [0, 0]
  #等高线 圈里面的都是1，外面的都是0
  c = contour(u, v, z, 0);

def ShowPictures(x):
  figure()
  x = x[np.random.permutation(len(x))[0:100]]
  for i in range(0,100):
    subplot(10, 10, i+1)
    axis('off')
    imshow(x[i].reshape(20,20).T, cmap='gray')

  show()

def ShowPicture(x, i):
#  show_data = x[np.random.permutation(5000)[0:100],:]
  axis('off')
  imshow(x[i].reshape(20,20).T, cmap='gray')
  show()

if __name__ == '__main__':
  #参数读取
  import sys
  import argparse
  parser = argparse.ArgumentParser(description = '逻辑回归')
  parser.add_argument('--train', help='训练文件')
  parser.add_argument('--label', help='标签文件(可选)')

  args = parser.parse_args()

  if args.train == None:
    parser.print_help()
    exit()

