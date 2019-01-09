'''
Now clustering is of two types flat and hierarichal
In flat clustering we told machine to make k number of clusters and in hierarichal clustering machine itself decide in how many clusters it want to do it
Problem with K_means(flat clustering) is it always try to cluster into groups of equal sizes and face problem ( i guess) why group size is not similar
'''
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
X=np.array([[1,2],
            [1.5,1.8],
            [5,8],
            [8,8],
            [1,0.6],
            [9,11]])

#plt.scatter(X[:,0],X[:,1],s=150)
#plt.show()

clf=KMeans(n_clusters=2)
clf.fit(X)

centroids=clf.cluster_centers_
labels=clf.labels_               #labels will be an array of value 0 and 1 which for each data point [0 0 1 1 0 1]
print(centroids)                 #[[1.16666667 1.46666667]
                                 #[7.33333333 9.        ]]
print(labels)                    #[0 0 1 1 0 1]
colors=["g.","r.","c.","b.","k.","o."]

for i in range(len(X)):
    plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=25)           #plotting X data

plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=150,linewidth=5)
plt.show()
