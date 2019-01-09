'''
                                                                    K nearest neighbour algorithm
->To classify our dataset, where we actually measures the euliadian distance of each point from all point and select minimum k distances and depending on that,
    we classify weather it belong to first class or second class..
'''
import numpy as np
from sklearn import preprocessing ,cross_validation,neighbors
import pandas as pd

df=pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace=True)             #don't want to loose data,hence outlier is being replaced as -99999
df.drop(['id'],1,inplace=True)                  #dropping redundant features eg-id , and  KNeighborsClassifier is very bad at handling redundant features

X=np.array(df.drop(['class'],1))
y=np.array(df['class'])

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)   #cross_validation shuffling 20% of our data and storing them in X_test y_test etc..

clf=neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)                        #training KNeighborsClassifier

accuracy=clf.score(X_test,y_test)               #testing classifier
print(accuracy)

# Now we want to do our prediction

example_measures=np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])

example_measures=example_measures.reshape(len(example_measures),-1)
prediction=clf.predict(example_measures)
print(prediction)
