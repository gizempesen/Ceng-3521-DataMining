
from sklearn import datasets 
from sklearn import linear_model 
from sklearn import multiclass 
from sklearn import model_selection 
from sklearn import metrics 
from sklearn import decomposition 
from sklearn.utils import resample 
from IPython.display import Image
import matplotlib.pyplot as plt
import warnings
#warnings.filterwarnings('Ignore')
import numpy
import time
#----------------------------------------------------------------------------------------------------

def task1(sample_size,dimension_size):
  X, y = datasets.make_sparse_uncorrelated(sample_size,dimension_size)
  X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y, test_size=0.3,train_size= 0.7)
  reg=linear_model.SGDRegressor(max_iter=1000)
  reg.fit(X_train,y_train)
  y_predict = reg.predict(X_test)
  error = metrics.mean_squared_error(y_test,y_predict)
  return error

  time_list = list()
  error_list = list()

  for r in [100,1000,2000]:
    for i in range(0,5):
      start = time.time()
      error_list.append(task1(10000,r))
      stop = time.time()
      time_list.append(stop - start)
    print(str(r) + ": Error:" + str(sum(error_list)/ len(error_list))+"-time:" + str(sum(time_list)))  

  for r in [100000,250000,500000]:
    for i in range(0,5):
      start = time.time()
      error_list.append(task1(10000,r))
      stop = time.time()
      time_list.append(stop - start)
    print(str(r) + ": Error:" + str(sum(error_list)/ len(error_list))+"-time:" + str(sum(time_list)))  


#----------------------------------------------------------------------------------------------------    
def task2_pca(sample_size,dimension_size,component_size):
  X,y = datasets.make_sparse_uncorrelated(sample_size,dimension_size)
  pca = decomposition.PCA(n_components=component_size)
  pca.fit(X)
  new_X = pca.transform(X) 
  X_train, X_test, y_train, y_test = model_selection.train_test_split(new_X, y, test_size =0.3, train_size = 0.7)
  start = time.time()
  reg = linear_model.SGDRegressor(max_iter=1000)
  reg.fit(X_train,y_train)
  y_predict = reg.predict(X_test)
  error = metrics.mean_squared_error(y_test, y_predict)
  stop = time.time()
  return error, stop-start

def task2_sampling(sample_size,dimension_size,new_sample_size):
  X,y = datasets.make_sparse_uncorrelated(sample_size,dimension_size)
  X_new, y_new = resample(X,y,n_samples = new_sample_size,replace=False)
  X_train, X_test, y_train, y_test = model_selection.train_test_split(X_new, y_new,test_size =0.3, train_size = 0.7)
  start = time.time()
  reg = linear_model.SGDRegressor(max_iter=1000)
  reg.fit(X_train,y_train)
  y_predict = reg.predict(X_test)
  error = metrics.mean_squared_error(y_test, y_predict)
  stop = time.time()
  return error, stop-start

error_list2 = list()
time_list2 = list()  

#for property c.
for r in [500,100,10,4,1]:
  for i in range(0,5):
    start = time.time()
    error_list2.append(task2_pca(1000,2000,r))
    stop = time.time()
    time_list2.append(stop - start)
  print(str(r) + ": Error:" + str(sum(error_list2)/ len(error_list2))+"-time:" + str(sum(time_list2)))  

  #for property c.
for r in [3000,1500,100,1000,100]:
  for i in range(0,5):
    start = time.time()
    error_list2.append(task2_sampling(5000,100,r))
    stop = time.time()
    time_list2.append(stop - start)
  print(str(r) + ": Error:" + str(sum(error_list2)/ len(error_list2))+"-time:" + str(sum(time_list2)))  
  
#----------------------------------------------------------------------------------------------------    
def task3():
  X,y = datasets.make_moons((100,100))
  X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size =0.3, train_size = 0.7)
  start = time.time()
  reg = linear_model.SGDClassifier(loss='log',max_iter=10000)
  reg.fit(X_train,y_train)
  y_predict = reg.predict(X_test)
  index = y_train == 1
  X_train_c1 = X_train[index,:]
  index = y_train == 0
  X_train_c2 = X_test[index,:]
  index = y_test == 1
  X_test_c1 = X_test[index,:]
  index = y_test == 0
  X_test_c2 = X_test[index,:]

  f,(ax1,ax2) = plt.subplots(1,2)
  ax1.plot(X_train_c1[:,0],X_train_c1[:,1],'ro')
  plt.hold(True)
  ax1.plot(X_train_c2[:,0],X_train_c2[:,1],'bx')
  ax2.plot(X_train_c1[:,0],X_test_c1[:,1],'ro')
  ax2.plot(X_train_c2[:,0],X_test_c2[:,1],'bx')
  ax1.set_title('Train')
  ax2.set_title('Test')
  ax1.set(xlabel='x1')
  ax1.set(ylabel='x2')
  ax2.set(xlabel='x1')
  ax2.set(ylabel='x2')
  f.set_size_inches(10,4)
  xmin, xmax = plt.xlim()
  ymin, ymax = plt.ylim()
  coef = reg.coef_
  intercept = reg.intercept_

  def line(x0):
    return (-(x0 * coef[0,0]) - intercept[0]) / coef[0,1]

  plt.plot([xmin, xmax],[line(xmin),line(xmax)], ls ="---",color='black')
  plt.legend(['class1','class2','hypothesis'])
  plt.show()

task3()  
