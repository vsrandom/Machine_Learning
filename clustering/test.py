import numpy as np

X=np.array([[1,2],
            [1.5,1.8],
            [5,8],
            [8,8],
            [1,0.6],
            [9,11]])

centroids={0:[1,2],1:[1.5,1.8]}

max_iter=300
k=2
for i in range(max_iter):
    classifications={}

    for i in range(k):
        classifications[i]=[]

    for features in X:
        distances=[np.linalg.norm(features-centroids[centroid]) for centroid in centroids]
        classification=distances.index(min(distances))
        classifications[classification].append(features)


print(classifications)
