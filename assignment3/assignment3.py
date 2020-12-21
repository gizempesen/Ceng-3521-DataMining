#Import library 
import warnings
import numpy as np
from sklearn import datasets 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import linear_model
warnings.filterwarnings("ignore") 
from sklearn import model_selection 
from sklearn.datasets import make_moons 
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#-----------------------
#2.1.1 Bagging & Pasting
#-----------------------
def task1():
#1. Load digit dataset (D).
#X,y = load_digits(return_X_y=True)
  digits = datasets.load_digits()
  X = digits.data
  y = digits.target

#2. 70% tuples are used for training while 30% tuples are used for testing. 
  X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y, test_size=0.3, train_size= 0.7, random_state=42) 
  scaler = StandardScaler()
  scaler.fit(X_train)
  X_train = scaler.transform(X_train)# Now apply the transformations to the data
  X_test = scaler.transform(X_test)

#3. Create an instance of multi-layer perceptron network 
  mlp = MLPClassifier(hidden_layer_sizes=(16,8,4,2),max_iter=1001) #(four hidden layers with 16, 8, 4,and 2 neurons in order) 
  mlp.fit(X_train,y_train)

#4. Apply bagging classifier  with eight base classifiers created at the previous step.
  clf = BaggingClassifier(mlp,n_estimators=8 )
  clf.fit(X_train, y_train)
  clf.score(X_test,y_test)

  predictions = mlp.predict(X_test)
  cm = confusion_matrix(y_test, predictions, labels=mlp.classes_)
  #print("Confusion Matrix:\n",cm,"\n") #confusion matrix
  accuracy = accuracy_score(y_test,predictions)
  #print("\nClassification Report:\n",classification_report(y_test,predictions),"\n")
  #print(accuracy)  
  predicted_instances_per_class = cm[np.eye(len(clf.classes_)).astype("bool")]

#6. Print your findings 
  estimators = clf.estimators_ 
  #print(len(estimators), type(estimators[0]))
  #pred_list = []
  #5. Calculate number of correctly classified test instance for each base classifier and finally for bagging classifier.

  #print(X_test.shape[1])
  #for base_estimator in estimators:
      #pred_list.append(base_estimator.predict(X_test)) 
      #print(X_test.shape[base_estimator])
  for i in predicted_instances_per_class:
    print(i, " out of 540 instances are correctly classified by learner")

  print("-------------------------------------------")  
task1()
#--------------
#2.1.2 Boosting
#--------------


def task2():
#2.Give Gaussian noise to D with a deviation value of 0.2.  
  noise = 0.2  

#1. Load moon dataset D with a tuple size greater than 100 .  
  X,y = make_moons(n_samples=100, noise=noise)

#3. 70% tuples are used for training while 30% tuples are used for testing.   
  X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size =0.3, train_size = 0.7)

#4. Create an instance of logistic regression algorithm with SGD solver 
  clf = LogisticRegression(random_state=0)
  clf.fit(X,y)
  reg = linear_model.SGDClassifier(loss='log',max_iter=10000) #(with ‘log’ loss function).
  reg.fit(X,y)

  #5. Apply AdaBoost classifier with four base classifiers created at the previous step
  clf_2 = AdaBoostClassifier(clf,n_estimators=100, random_state=0)
  clf_2.fit(X, y)

#6. Through a 1×4-axis figure, visualize testing samples

#--------------------------------
#2.2 Different learning algorithm
#--------------------------------


def task3():
#1. Load breast-cancer dataset D 
  data = load_breast_cancer()
  X = data.data
  y = data.target
  X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y, test_size=0.3, train_size= 0.7, random_state=42) 

#2. Create instances of three classification algorithms (feel free to choose estimator algorithms).
  log_clf = LogisticRegression()
  rnd_clf = RandomForestClassifier()
  svm_clf = SVC()
  voting_clf = VotingClassifier(
    estimators=[('lr', log_clf),('rf',rnd_clf),('svc',svm_clf)],voting='hard')
  voting_clf.fit(X_train, y_train)

#4. Calculate accuracy of every base classifier and ensemble classifier  
  for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
#5. Print    
    print("Accuracy obtained by learner",clf.__class__.__name__ + " is:")
    print(accuracy_score(y_test, y_pred))

task3()    

#-------------------------------  
#2.3 Different parameter setting
#-------------------------------

def task4():
#1. Load breast-cancer dataset (D).
  data = load_breast_cancer()
  X = data.data
  y = data.target

#3. Create 10 instances of multi-layer perceptron network with different hidden layer and neuron size.    
  hidden_size = 10
  loss_train = np.zeros(hidden_size)
  loss_test = np.zeros(hidden_size)

#2. 70% tuples are used for training while 30% tuples are used for testing.   
  X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y, test_size=0.3,train_size= 0.7, random_state=42)

#4
  for i in range(1, hidden_size +1):
    a = tuple() #hidden layer size
    for t in range(1, i+1):
      a = (2 ** (t),) + a #until 2**n
    mlp = MLPClassifier(a, activation="relo", solver = "sgd", shuffle =True)
    mlp.fit(X_train, y_train)
    #y_pred = mlp.predict(X_test)
    #print(accuracy_score(y_test, y_pred))
    loss_train[i - 1] = mlp.score(X_train,y_train) #with score we calculate loss value
    loss_test[i - 1] = mlp.score[X_test, y_test]

#task4()
  

#--------------------------------
#3 k-Nearest Neighbors Classifier
#--------------------------------


#2. Give Gaussian noise to D with a deviation value of 0.3.
n= 0.3

#1. Load moon dataset D with a tuple size greater than 100 .
X,y = make_moons(n_samples=100, noise=n) 

#3. 70% tuples are used for training while 30% tuples are used for testing.    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#4. Apply k-NN classifier for each testing tuples 
neigh = KNeighborsClassifier(n_neighbors=5)#(with k = 5 setting).
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)

#5. Through a 1×4-axis figure. For each axis, visualize training samples
def get_neighbors(xs, sample, k=5):
  neighbors = [(x, np.sum(np.abs(x - sample))) for x in xs]
  neighbors = sorted(neighbors, key=lambda x: x[1])
  return np.array([x for x, _ in neighbors[:k]])

_, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
for i in range(4):
  sample = X_test[i]
  neighbors = get_neighbors(X_train, sample, k=5)
  ax[i].scatter(X_train[:, 0], X_train[:, 1], c="skyblue")
  ax[i].scatter(neighbors[:, 0], neighbors[:, 1], edgecolor="green")
  ax[i].scatter(sample[0], sample[1], marker="+", c="red", s=100)
  ax[i].set(xlim=(-2, 2), ylim=(-2, 2))

  plt.tight_layout()
    

