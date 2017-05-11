#Author: Steven Seung
#Example of KMeans algoritm upon the Iris Data Set

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

#organize numpy arrays into arrays according to training data y
def organize_data(x,y, n_samples, n_groups):
    xax = []
    yax = []
    for i in range(n_groups):
        xax.append([])
        yax.append([])

    for i in range(n_samples):
        for n in range(n_groups):
            if y[i]==n:
                xax[n].append(x[i][0])
                yax[n].append(x[i][1])
    
    return xax, yax

#plotting function shows groups with kmean estimated centers
def plot_data(x,data,km,xlabl='',ylabl=''):
    plt.figure()
    x_col, y_col = organize_data(x,data.target,x.shape[0],3)

    for i in range(3):
        plt.scatter(x_col[i],y_col[i])

    plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1], marker='X')
    plt.xlabel(xlabl)
    plt.ylabel(ylabl)

#get iris data
data = datasets.load_iris()

#pair up data features
#0:Sepal Length 1:Sepal Width 2:Pedal Length 3: Petal Width
X01 = data.data[:,:2]
X23 = data.data[:,2:]
X02 = np.column_stack((data.data[:,0],data.data[:,2]))
X13 = np.column_stack((data.data[:,1],data.data[:,3]))

#training via KMeans
#guess the number of clusters to be 3 
print('begin training')
km01 = KMeans(n_clusters=3, n_init = 20).fit(X01)
km23 = KMeans(n_clusters=3, n_init = 20).fit(X23)
km02 = KMeans(n_clusters=3, n_init = 20).fit(X02)
km13 = KMeans(n_clusters=3, n_init = 20).fit(X13)
print('end training')
#plotting
plot_data(X01,data,km01, 'sepal length', 'sepal width')
plot_data(X23,data,km23, 'petal length', 'petal width')
plot_data(X02,data,km02, 'sepal length', 'petal length')
plot_data(X13,data,km13, 'sepal width', 'petal width')

plt.show()

#print(km23.predict([4.2,1.0]))



