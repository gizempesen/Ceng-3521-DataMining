# import statements
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys, os


def task1_original_data():

# create blobs
  data = make_blobs(n_samples=200, n_features=6, centers=6, cluster_std=1.6, random_state=50)
# create scatter plot
  plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='jet',marker="+",label="Original Data")
  plt.xlim(-15,15)
  plt.ylim(-15,15)  
  plt.show(block=False)
  plt.pause(3)
  plt.close()
  
def task1_iterations_print():
# create blobs
  data = make_blobs(n_samples=200, n_features=6, centers=6, cluster_std=1.6, random_state=50)

#Finding iteration points(updated)
  print("First iteration points:")  
  kmeans = KMeans(n_clusters=6,random_state=0,max_iter=1)
  kmeans.fit(data[0])
  print(kmeans.cluster_centers_)
  print("Second iteration points:")
  kmeans = KMeans(n_clusters=6,random_state=0,max_iter=2)
  kmeans.fit(data[0])
  print(kmeans.cluster_centers_)
  print("Third iteration points:")
  kmeans = KMeans(n_clusters=6,random_state=0,max_iter=3)
  kmeans.fit(data[0])
  print(kmeans.cluster_centers_)
  print("Forth iteration points:")
  kmeans = KMeans(n_clusters=6,random_state=0,max_iter=4)
  kmeans.fit(data[0])
  print(kmeans.cluster_centers_)

#task1_iterations()

def task1_iterations_view():

  data = make_blobs(n_samples=200, n_features=8, 
                           centers=6, cluster_std=1.8,random_state=101)
  data[0].shape
 
  fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(10,10))

  c=d=0
  for i in range(4):
      ax[c,d].title.set_text(f"{i+1} iteration points:")
      kmeans = KMeans(n_clusters=6,random_state=0,max_iter=i+1)
      kmeans.fit(data[0])
      centroids=kmeans.cluster_centers_
      ax[c,d].scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='jet')
      ax[c,d].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black')
      d+=1
      if d==2:
          c+=1
          d=0  
  for i in range(4):
    plt.show(block=False)
    plt.pause(3)
    plt.close()       
  
        

def task2_original_data():
  # create dataset
  X, y = make_blobs(
     n_samples=150, n_features=2,
    centers=3, cluster_std=0.5,
    shuffle=True, random_state=0)

  # plot
  plt.scatter(
     X[:, 0], X[:, 1],
     c='white', marker='o',
     edgecolor='black', s=50  
  )
  plt.show(block=False)
  plt.pause(3)
  plt.close()  

def task2_cluseters():

  # create dataset
  X, y = make_blobs(
  n_samples=150, n_features=2,
  centers=3, cluster_std=0.5,
  shuffle=True, random_state=0)


  km = KMeans(
   n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
  )
  y_km = km.fit_predict(X)


  # plot the 3 clusters
  plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1')

  plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2')

  plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3')

  # plot the centroids
  plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids')
  plt.legend(scatterpoints=1)
  plt.grid()
  plt.show(block=False)
  plt.pause(3)
  plt.close()



#------------------------------------------
os.system("clear")
#On the Screen ..
print("------------------------------------------------------------")
print("Gizem PESEN 170709050\n ")
print("Welcome Assignment4 Homework\n ")
print("screens will close after 3 sec.\n ")
print("------------------------------------------------------------")
print("\n \n  Please enter the command number to execute the script: \n"
      " 1 : Original Data in Task1\n"
      " 2 : Write the iteration points\n"
      " 3 : Steps in K-means \n"
      " 4 : Original Data in Task2\n"
      " 5 : Convergence of centroids\n"
      " 0 : Exit the script.\n")
print("------------------------------------------------------------\n")
while(True):
  command = input("\nPlease enter the command number:\n")
  if(command==None or command=="" or command==" "):     #base case
    print("Unvalid command number was sended.")
  
  elif(command=="1"):
    task1_original_data()

  elif(command=="2"):
    task1_iterations_print() 

  elif(command=="3"):
    task1_iterations_view() 

  elif(command=="4"):
    task2_original_data()  

  elif(command=="5"):
    task2_cluseters()   

  elif(command=="0"):
    print("Exiting from the script.")
    sys.exit()
      
    