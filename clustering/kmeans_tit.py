#Now we will ask K_means, Hey K_means take this dataset and seprate it into groups
#Now we want to see if K_means would be able to seprate them into two groups ie people who died and people who lived!!
#(Not confirm) If we could predict out a new feature using clustering after training the classifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd


df=pd.read_excel('titanic.xls')
#print(df.head())
df.drop(['body','name'],1 ,inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)


def handel_non_numerical_data(df):
    columns=df.columns.values

    for column in columns:
        text_digit_vals={}      #empty dictionary ,it will be something like {'Female':0,'male':1..}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype!= np.int64 and df[column].dtype!=np.float64: #checking if column is on int or float type or not
            column_contents=df[column].values.tolist()   #converting each column into a list and storing them in column_contents
            unique_elements=set(column_contents)   #give all unique non repetative values
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique]=x
                    x+=1

            df[column]=list(map(convert_to_int,df[column])) #again setting the values into the previous columns!!
    return df

df=handel_non_numerical_data(df)
X=np.array(df.drop(['survived'],1).astype(float))
X=preprocessing.scale(X)
y=np.array(df['survived']) ## [1 1 0 ... 0 0 0]

clf=KMeans(n_clusters=2)
clf.fit(X)

correct=0
for i in range(len(X)):
    predict_me=np.array(X[i].astype(float))
    predict_me=predict_me.reshape(-1,len(predict_me))
    prediction=clf.predict(predict_me)
    if prediction[0]==y[i]:                          #now we know KMeans will be dividing groups into two clusters and each data point will be either 0 or 1,hence predict_me is always going to be zero or one. Now K_means don't know that we are expecting it to cluster it into groups ,of survival and non-survival hence if accuracy comes out to be 20% consistently,then it is 80%, because we can flip it!!(I hope it is making sense)
        correct+=1

print(correct/len(X))
#handel_non_numerical_data(df)
#print(df.head())
