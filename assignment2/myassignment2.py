
#Import library 
import time 
import warnings
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import model_selection 
warnings.filterwarnings("ignore")
from sklearn import linear_model #for linear regression
from sklearn.datasets import make_classification #classification
import numpy as np #single layer neural network 
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report,confusion_matrix

#-----------------------------------------------------------------------------
#2.1 Classification task

def task1(sample_size,dimension_size,n_iter):
  X, y = make_classification(sample_size,dimension_size)
  X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y, test_size=0.3,train_size= 0.7)
  clf=linear_model.SGDClassifier(loss="log",max_iter=n_iter)
  clf.fit(X_train,y_train)

  scaler = StandardScaler()
  scaler.fit(X_train)
  X_train_std = scaler.transform(X_train)
  X_test_std = scaler.transform(X_test)
  np.unique(y)
  X_train_std =  X_train_std[:,[2,3]]
  X_test_std = X_test_std[:,[2,3]]
  
  ppn = Perceptron(max_iter=n_iter,eta0 =0.1)
  ppn.fit(X_train_std,y_train)

  #first I find error 
  y_predict = ppn.predict(X_test_std)
  error = metrics.mean_squared_error(y_test,y_predict)
  return error

  #then I tried with accuracy score
  #y_predict = ppn.predict(X_test_std)
  #accuracy = accuracy_score(y_test,y_predict)*100
  #return accuracy

time_list = list()
error_list = list()

for r in [100,1000]:
  for i in range(0,10):
    start = time.time()
    error_list.append(task1(10000,r,500)) 
    stop = time.time()
    time_list.append(stop - start)
  print(str(r) + ": Error:" + str(sum(error_list)/ len(error_list))+" -Time:" + str(sum(time_list)))  


#-----------------------------------------------------------------------------
#2.2 Visualization of decision boundary
def task2():

  input_features = np.array([[0,0],[0,1],[1,0],[1,1]]) #define input features

#  X, y = make_classification(sample_size,dimension_size)
  #X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y, test_size=0.3,train_size= 0.7)

  iris = datasets.load_iris()
  X = iris.data[:, :3]  # we only take the first three features.
  Y = iris.target

  #make it binary classification problem
  X = X[np.logical_or(Y==0,Y==1)]
  Y = Y[np.logical_or(Y==0,Y==1)]

  model = svm.SVC(kernel='linear')
  clf = model.fit(X, Y)
  z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) /   clf.coef_[0][2]
  tmp = np.linspace(-5,5,30)
  x,y = np.meshgrid(tmp,tmp)
  fig = plt.figure()
  ax  = fig.add_subplot(111, projection='3d')
  ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'ob')
  ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'sr')
  ax.plot_surface(x, y, z(x,y))
  ax.view_init(30, 60)
  return plt.show()

task2()  

#-----------------------------------------------------------------------------
#3.1 Error convergence with multi-layer perceptron
def task3():
  digits = datasets.load_digits()
  X_train,X_test,y_train,y_test = model_selection.train_test_split(digits,digits.target, test_size=0.3,train_size= 0.7)
  scaler = StandardScaler()
  scaler.fit(X_train)
  # Now apply the transformations to the data:
  X_train = scaler.transform(X_train)
  X_test = scaler.transform(X_test)
  mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
  mlp.fit(X_train,y_train)
  predictions = mlp.predict(X_test)
  confusion_matrix(y_test,predictions)
  print(classification_report(y_test,predictions))

task3()
