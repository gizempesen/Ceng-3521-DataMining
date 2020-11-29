
#--------------------------------------------------------------------
#Import library 
#--------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import make_blobs #generate data
from matplotlib import pyplot #data visuation
import matplotlib.pyplot as plt
from pandas import DataFrame #data visuation
import numpy as np #to train and test
from sklearn.model_selection import train_test_split #to train and test
from sklearn.datasets import make_classification #classification
from sklearn.datasets import make_regression #classification
from sklearn.linear_model import SGDRegressor #Sgd
from sklearn import linear_model #for linear regression
import time #calculate training time
import numpy as np #for pca
from sklearn.decomposition import PCA #pca
from sklearn.metrics import mean_squared_error #to calculate error


#--------------------------------------------------------------------
#                  2.1 Curse of dimensionality



#--------------------------------------------------------------------
# generate 2d classification dataset
#--------------------------------------------------------------------
m = 500
n = 1

X, y = make_regression(n_samples=m, n_features=n, noise=0.1)

#--------------------------------------------------------------------
# training and testing
#--------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#--------------------------------------------------------------------
#  linear regression with (SGD) with 1,000 iterations
#--------------------------------------------------------------------

n_iter=1000 #with 1,000 iterations

start_time = time.time() #calculate training time
clf = SGDRegressor(max_iter=n_iter)
clf.fit(X_train,y_train)#train the model or fits a linear model
prediction_test = clf.predict(X_test)  # make a prediction
print("--- %s seconds1 ---" % (time.time() - start_time))
print("Error1 =",mean_squared_error(y_test,prediction_test))



#--------------------------------------------------------------------
#             2.2 Sampling and dimensionality reduction

#----------------------------------------------------
# apply Principal Component Analysis (PCA)  
#----------------------------------------------------
pca = PCA(n_components=2)
pca.fit(X)
#print(pca.explained_variance_ratio_)
#print(pca.singular_values_)

#--------------------------------------------------------------------
# training and testing
#--------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#--------------------------------------------------------------------
#  linear regression with (SGD) with 1,000 iterations
#--------------------------------------------------------------------

n_iter=1000 #with 1,000 iterations

start_time = time.time() #calculate training time
clf = SGDRegressor(max_iter=n_iter)
clf.fit(X_train,y_train)   #Train the model or fits a linear model
prediction_test = clf.predict(X_test)  # make a prediction
print("--- %s seconds2 ---" % (time.time() - start_time))
print("Error2 =",mean_squared_error(y_test,prediction_test))

#--------------------------------------------------------------------
# Visualization 
#--------------------------------------------------------------------
#Plot would be useful for lot of data points
#plt.scatter(prediction_test, prediction_test-y_test)
#plt.hlines(y=0, xmin=200, xmax=300)
#plt.show()   

#--------------------------------------------------------------------
#             3.1 Visualization and binary-class classification
#--------------------------------------------------------------------
from sklearn.datasets import make_moons
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression 

#---------------------------------------------------------------------
#Load a dataset from sklearn module (e.g., moon, circle, and the like) 
#---------------------------------------------------------------------
X, y = make_moons(n_samples=100, noise=0.1)

#--------------------------------------------------------------------
# training and testing
#--------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#---------------------------------------------------------------------
#logistic regression algorithm with SGD solver
#---------------------------------------------------------------------
clf = LogisticRegression().fit(X_test[:100], y_test[:100])

#---------------------------------------------------------------------
#Through a 1×2-axis figure, visualize training and testing samples as well as a decision boundary
#---------------------------------------------------------------------

xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
grid = np.c_[xx.ravel(), yy.ravel()]
probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)


f, ax = plt.subplots(figsize=(8, 6))
ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)

ax.scatter(X[:,0], X[:, 1], c=y, s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)

ax.set(aspect="equal",
       xlim=(-5, 5), ylim=(-5, 5),
       xlabel="$X_1$", ylabel="$X_2$")


plt.show()   

#--------------------------------------------------------------------
#             3.1 Visualization and binary-class classification
#-------------------------------------------------------------------- 

from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)
