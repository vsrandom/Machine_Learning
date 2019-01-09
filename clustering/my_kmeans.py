import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X=np.array([[1,2],
            [1.5,1.8],
            [5,8],
            [8,8],
            [1,0.6],
            [9,11],
            [1,3],
                                [8,9],
                                [0,3],
                                [5,4],
                                [6,4]])


plt.scatter(X[:,0],X[:,1],s=150)
plt.show()

colors=["g","r","c","b","k"]

class K_Means:
    def __init__(self,k=2,tol=0.001,max_iter=300):
        self.k=k
        self.tol=tol
        self.max_iter=max_iter

    def fit(self,data):

        self.centroids={}

        for i in range(self.k):
            self.centroids[i]=data[i]         #dictionary self.centroids={0:[1,2],1:[1.5,1.8]}

        for i in range(self.max_iter):
            self.classifications={}         #now everytime we will be clearing our classification dictionary as  our centroids will shift

            for i in range(self.k):
                self.classifications[i]=[]  #classifications={0:,1:}

            for features in data:
                distances=[np.linalg.norm(features-self.centroids[centroid]) for centroid in self.centroids] #[x1,x2] for each point in data
                #where x1 is distance of point from ist centroid and x2 is distance from 2nd centroid
                classification=distances.index(min(distances)) #it will be either 0 or 1
                self.classifications[classification].append(features)#filling our dictionary when centroids were [1,2] and [1.5,1.8]
                #{0: [array([1., 2.])], 1: [array([1.5, 1.8]), array([5., 8.]), array([8., 8.]), array([1. , 0.6]), array([ 9., 11.])]}

            prev_centroids=dict(self.centroids) #for comparison

            for classification in self.classifications:
                self.centroids[classification]=np.average(self.classifications[classification],axis=0)
                optimized=True

            for c in self.centroids:
                original_centroid=prev_centroids[c]
                current_centroid=self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100)>self.tol:
                    optimized=False

            if optimized:
                break

    def predict(self,data):
        distances=[np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids] #[x1,x2] for each point in data
        #where x1 is distance of point from ist centroid and x2 is distance from 2nd centroid
        classification=distances.index(min(distances)) #it will be either 0 or 1
        return classification


clf=K_Means()
clf.fit(X)


for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1],marker="o",color='k',s=150,linewidth=5)

for classification in clf.classifications:
    color=colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1],marker='x',color=color,s=150,linewidth=5)

'''
unknowns=np.array([[1,3],
                    [8,9],
                    [0,3],
                    [5,4],
                    [6,4]])


for unknown in unknowns:
    classification=clf.predict(unknown)
    plt.scatter(unknown[0],unknown[1],marker="*",color=colors[classification],s=150,linewidth=5)
'''
plt.show()
