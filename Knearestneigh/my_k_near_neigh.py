import numpy as np
from math import sqrt
import warnings
import pandas as pd
import random
from collections import Counter                 #will act as counter for k

def k_nearest_neighbors(data,predict,k=3):
    if(len(data)>=k):
        warnings.warn('K is set to a value less than total voting groups')
    distances=[]
    for group in data:
        for features in data[group]:
            euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict))#it is actually the built in function from numpy which allow us,#to calculate euclidean_distance for n dimension features(#cheating)
            distances.append([euclidean_distance,group]) #distances is a list of lists [[a,b]..] a is distance from each element of group k and r, b is 1 or 2
    #print(distances) #-> [[6.4031242374328485, 'k'], [5.0, 'k'], [6.324555320336759, 'k'], [2.23606797749979, 'r'], [2.0, 'r'], [3.1622776601683795, 'r']]
    votes=[i[1] for i in sorted(distances)[:k]] # distances sorted by their forst value i.e the distances and we only need first k values from starting
    #print(votes) # -> ['r', 'r', 'r']
    #print(Counter(votes).most_common(1)) #->[('r', 3)], now most_common(1) means select only one which is the most common ie r (which is 3 times) here
    #in case we hade most_common(2) then our output will be have two list elements(like take new_features=5,4  and result would be [('k', 2), ('r', 1)]
    vote_result=Counter(votes).most_common(1)[0][0] #now two zeroes meaning is clear
    return vote_result


df=pd.read_csv("breast-cancer-wisconsin.data.txt")
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)

full_data=df.astype(float).values.tolist() # because some of data was like being treated as string ,hence we converted everything in float and stored in list of lists
#print(full_data[:5])
#print(20*'#')
random.shuffle(full_data)#now we are shuffeling that list of lists and each entry in list contain values till label values
#print(full_data[:5])

test_size=0.2
train_set={2:[],4:[]}  #dictionary of classes
test_set={2:[],4:[]}
train_data=full_data[:-int(test_size*len(full_data))]  #initial 80% of data
test_data=full_data[-int(test_size*len(full_data)):]   #final 20% of data

#populating our dictionaries
for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

#print(train_set)
#print(test_set)
'''
print(20*'#')
for group in train_set:
    for features in train_set[group][:5]:
        print(features)

'''
correct=0
total=0

#print('$')
'''
Code similar to what we are doing!!!
x={2:[[1,2,3,4,5,6],[23,43,3,12,45,43]], 4:[[21,2,3,4,1,34],[28,34.66,90,100,45]]}
print(x)
print(20*'#')
for group in x:
    for data in x[group]:
        print(data)
'''

for group in test_set:
    for datax in test_set[group]:
        vote=k_nearest_neighbors(train_set,datax,k=5)  #because by default k is 5 in sicket learn module and since we are doing comparison bw our code and scklearn classifier
        if group==vote:
            correct+=1
        total+=1

print('accuracy:',correct/total)

#print(float(c)/float(t))
#print(float(t))
