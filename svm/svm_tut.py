# ..........................................................SVM is also a ML classifier or algorithm.......................................................................
'''
It is basiclly a binary classifier but it can be used for multi classification as well and in this what we basicslly do is,we have a dataset say +ve and -ve points
and whe make a decision boundary or a seprate plane form which the perpendicular distance of each point is maximum,and in prediction depending on where the point is with respect to the decision boundary ,we classify it
'''


import numpy as np
from sklearn import cross_validation,preprocessing,svm
import pandas as pd

df=pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)

X=np.array(df.drop(['class'],1))
y=np.array(df['class'])

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

clf=svm.SVC()
clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)
print(accuracy)

example=np.array([[4,1,1,1,2,2,3,4,1],[4,2,1,2,2,2,3,2,1]])
example=example.reshape(len(example),-1)

prediction=clf.predict(example)
print(prediction)
