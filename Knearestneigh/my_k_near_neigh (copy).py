import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter                 #will act as counter for k
style.use('fivethirtyeight')

dataset={'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]} #dictionary containg two classes k,r
new_features=[5,7]

'''
[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset] #plotting each element of each class
plt.scatter(new_features[0],new_features[1])
plt.show()
'''

def k_nearest_neighbors(data,predict,k=3):
    if(len(data)>=k):
        warnings.warn('K is set to a value less than total voting groups')
    distances=[]
    for group in data:
        for features in data[group]:
            euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict))#it is actually the built in function from numpy which allow us,#to calculate euclidean_distance for n dimension features(#cheating)
            distances.append([euclidean_distance,group]) #distances is a list of lists [[a,b]..] a is distance from each element of group k and r, b is 1 or 2
    print(distances) #-> [[6.4031242374328485, 'k'], [5.0, 'k'], [6.324555320336759, 'k'], [2.23606797749979, 'r'], [2.0, 'r'], [3.1622776601683795, 'r']]
    votes=[i[1] for i in sorted(distances)[:k]] # distances sorted by their forst value i.e the distances and we only need first k values from starting
    print(votes) # -> ['r', 'r', 'r']
    print(Counter(votes).most_common(1)) #->[('r', 3)], now most_common(1) means select only one which is the most common ie r (which is 3 times) here
    #in case we hade most_common(2) then our output will be have two list elements(like take new_features=5,4  and result would be [('k', 2), ('r', 1)]
    vote_result=Counter(votes).most_common(1)[0][0] #now two zeroes meaning is clear
    return vote_result


result=k_nearest_neighbors(dataset,new_features,k=3)
print(result)


[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset] #plotting each element of each class
plt.scatter(new_features[0],new_features[1], color=result)
plt.show()
