#!/usr/bin/python
#coding:utf-8

import numpy as np
#训练数据分类两个特征扩展多个高维特征
def MapFeature(x1, x2):
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

def hypothesis(x, theta):
  return sigmoid(np.dot(x ,theta))
  

def sigmoid(x):
  return 1 / (1 + np.exp(-1 * x))

def cost(x, y, theta, alpha, lam):
  #m records
  m = len(x)

  #
  h = hypothesis(x, theta)
  #print 'hypothesis',h
  #cost function
  vector_j = y * np.log(h) + (1 - y) * np.log(1 - h)
  #print 'vector_j',vector_j

  #regular item
  cost_j = -1.0 / m * np.sum(vector_j) + 1.0 / (2 * m) * np.dot(theta,theta)

  return cost_j

def gradient(x, y, theta, lam):
  m = len(x)

  h = hypothesis(x, theta)

  grad = np.zeros(len(theta), dtype = np.float)

  #theta zero
  grad[0] = 1.0 / m * np.sum((h - y) * x[:,0])

  for i in range(1, len(theta)):
    grad[i] = 1.0 / m * np.sum((h-y) * x[:,i]) + lam / m * theta[0]

  return grad

def GradientDescent(x, y, theta, alpha, lam, num_iters):
  
  cost_j = np.zeros(num_iters, dtype = np.float)
  cost_j[0] = cost(x, y, theta, alpha, lam)
  
  for i in range(1, num_iters):
    #梯度下降
    grad = gradient(x, y, theta, lam)
    theta1 = theta - grad
    cost_j[i] = cost(x, y, theta1, alpha, lam)

    #判断是否到达最小值
    if cost_j[i] - cost_j[i-1] >= 0:
      break
    theta = theta1  
    print 'iter %d, loss:%f' %(i, abs(cost_j[i] - cost_j[i-1]))
    #print 'cost:',cost_j[i]
    

    #当前theta的梯度值

  return (theta, cost_j[i-1])

def Predict(x, theta):
  map_x = MapFeature(x[:,0], x[:,1])
  return hypothesis(map_x, theta)


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
        map_x = MapFeature(u[i], v[j])
        #print 'map_x',map_x
        z[i,j] = np.dot(map_x, theta)

  # important to transpose z before calling contour
  #z = z.T

  # Plot z = 0
  # Notice you need to specify the range [0, 0]
  #等高线 圈里面的都是1，外面的都是0
  c = contour(u, v, z, 0);

#main function
def main():
  #参数读取
  import sys
  import argparse
  parser = argparse.ArgumentParser(description = '逻辑回归')
  parser.add_argument('--train', help='训练文件')
  
  args = parser.parse_args()
  
  if args.train == None:
    parser.print_help()
    exit()
  
  #
  trainfile = args.train
  print '训练文件:%s' %(trainfile)
  
  train_data = np.loadtxt(trainfile, delimiter = ',', dtype = object)
  #print 'train_data',train_data
  
  train_x = np.array(train_data[:,[0,1]], dtype = np.float)
  train_y = np.array(train_data[:,2], dtype = np.int)
  #print 'train_x', train_x
  #print 'trian_y', train_y
  
  #real train_x
  train_x_map = MapFeature(train_x[:, 0], train_x[:,1])
  #print 'MapFeature x:',train_x.shape
  #print 'mapfeature x:', train_x
  
  #theta
  theta = np.zeros(train_x_map.shape[1], dtype = np.float)
  #print 'theta',theta
  
  #number of iter
  max_num_iters = 500
  
  #learning rate
  alpha = 0.01
  
  #regularization
  lam = 1
  
  (last_theta, last_cost) = GradientDescent(train_x_map, train_y, theta, alpha, lam, max_num_iters)
  
  print '训练得到的theta:', last_theta
  print '最小的耗散:',last_cost
  
  #预测(训练数据)
  pre_y = Predict(train_x, last_theta)
  print '准确度为:%g' %(float(sum(train_y == (pre_y >= 0.5))) / len(train_y))
  
  
  #画图
  figure()
  #title('逻辑回归（正则化，二维转高维达到线性可分）')
  title('logistic regression.\n(tranform low dimensions to high dimensions)')
  xlabel('x')
  ylabel('y')
  PlotData(train_x, train_y)
  #画等高线
  PlotBoundary(last_theta)
  
  #第三个标签打不出来
  legend(['y = 1', 'y = 0', 'decision boundary']) 
  show()


if __name__ == '__main__':
  main()

