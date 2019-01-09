####..........................................PRACTICAL MACHINE LEARNING BY SENTDEX in Python ..................................................


'''
Regression- Take continous data and figoure out best regression line for that
Features and Labels
feastures are like attributes
Now in our data set,each column is a feature eg- Open High low etc
Now we only want meaningful features like we want to do pattern recognition we dont need
all features
Adj. open , adj high are after stock splits
adj.high - adj.low give volatility for that day
WTF is classifier?
lABEL GIVES SOME SORT OF PREDICTION IN FUTURE

'''

'''
To begin, what is regression in terms of us using it with machine learning? The goal is to take continuous data, find the equation that best fits the data, and be able
forecast out a specific value. With simple linear regression, you are just simply doing this by creating a best fit line.

From here, we can use the equation of that line to forecast out into the future, where the 'date' is the x-axis, what the price will be.

A popular use with regression is to predict stock prices. This is done because we are considering the fluidity of price over time, and attempting to forecast the next
fluid price in the future using a continuous dataset.

Regression is a form of supervised machine learning, which is where the scientist teaches the machine by showing it features and then showing it was the correct answer is,
over and over, to teach the machine. Once the machine is taught, the scientist will usually "test" the machine on some unseen data, where the scientist still knows what
the correct answer is, but the machine doesn't. The machine's answers are compared to the known answers, and the machine's accuracy can be measured. If the accuracy is
high enough, the scientist may consider actually employing the algorithm in the real world.

'''

import pandas as pd
import quandl , math, datetime
import numpy as np
#numpy will allow us to use array
from sklearn import preprocessing, cross_validation, svm
#cross_validation shuffles our data,pre-preprocessing for scaling
#we can use SVM to use regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle #a file to save the trained classifier,as we don't want to train our classifier everytime we run the programe it will be helpful when we will have
#GB's of data and it will save time

style.use('ggplot')
#to plot graphs of type ggplot , I guess not confirm!!!!!!!

df= quandl.get('WIKI/GOOGL') #data set from internet
#print(df.head())
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close']*100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open']*100.0
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
#print(df.head())


#.................IMPORTANT note -> We have data from 2004-08-19(19 august) to 2018-03-27 (27 march) and note that there are some days missing in it..................................

forecast_col = 'Adj. Close'
#print(df.head()) ###
#print(df.tail()) ###
df.fillna(-99999,inplace=True)
#print(df.head()) ###
#print(df.tail()) ###

forecast_out = int(math.ceil(0.01*len(df)))
#print(0.01*len(df))=34.24 and by cieling function 35
#print(forecast_out)=35

df['label'] = df[forecast_col].shift(-forecast_out)
'''
This is how I understood sentdex logic, he is taking 0.01 or 1% of the length of all the rows within the dataframe. Each row in the dataFrame is representative of a day in
the life of the stock. So if the stock has been trading for 365 days, there will be 365 rows in the dataFrame. 1% of 365 is 3.65 days which is then rounded up by the
math.ceil function to 4 days. The 4 days will be the forecast _out variable which is the variable that used to shift the Adj.Close price column in the dataFame up by 4.
In other words, if you were standing at day 1 of the stock when it was first traded, the prediction or the 'label' from his algorithm would tell you that at day 4, your
stock will be valued at the amount of the close as taken on day 4 from actual data. This isn't totally useful information since you can look at the Adj.Close column on day
4 to get back to the label info on day 1. This is really all done to build a training set so that the machine can learn from the trend

example of shift operation - https://stackoverflow.com/questions/20095673/shift-column-in-pandas-dataframe-up-by-one#

Q.probably silly of me but ....... if label was just shifted Adj. Close by some forecast_out .......... then how come the tail of label have values.
My Question basically :- if we shift a column upwards/downwards will it not lose values from top/bottom and gain NA s at bottom/top . how come label has values at both
top and bottom?﻿

Ans-The df.dropna() method removes the rows which contains NAs. So if we shift the label by 5 days, the last 5 days will be removed from the dataset. When you
print the tail of df, it won't show the removed days.﻿


'''
#print(df.head())                                                                                               #comment it
#print(df.tail())                                                                                               #comment it

#df.dropna(inplace = True)
'''>>> Since we have shifted the label by 0.01 of length of data frame ie 35 days,
lable have NA values at the end of 35 days ie from 2018-02-06 to 2018-03-27 and it drops the end values
'''
#print(df.head())                                                                                               #comment it
#print(df.tail(35))
                                                                                                                #comment it

#This course is paused, now doing ML course by Andrew NG

######.............................................CONTINUING THE SENTDEX COURSE........................................................................




X=np.array(df.drop(['label'],1)) #numpy allow us to use array,storing values in array
#print(X)
#y=np.array(df['label'])
X=preprocessing.scale(X)        #scaling each value in X array, somewhat ambigius
#print(X)
X_lately=X[-forecast_out:]      #last 35 days dataframe values stored in the X_lately ie there labels have NA values
#print(X_lately)
X=X[:-forecast_out]             #first 35 values of features stored in array ,remember this array has 4 colums
#print(X)

'''
............................................SLICING of arrays....................................................................................................
https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
# simple slicing
from numpy import array
# define array
data = array([11, 22, 33, 44, 55])
print(data[0:1])


[11]


# simple slicing
from numpy import array
# define array
data = array([11, 22, 33, 44, 55])
print(data[-2:])


[44 55]
'''

df.dropna(inplace=True) # last 35 rows are dropped and now our dataframe is from 2004-8-19 to 2018-02-05(ie 5th feb)
y=np.array(df['label']) # now y stores the label values in a 1 column array till 5th feb 2018
#print(len(X),len(y))
X_train,X_test,y_train,y_test= cross_validation.train_test_split(X,y,test_size=0.2)
#it will shuffle our data with lables and will give 20% of it to each of X_train,X_test,y_train,y_test
'''
clf=LinearRegression(n_jobs=-1) #Using linear regression from sicketlearn module
#clf=LinearRegression(n_jobs=x), allow us to do x instructions at a time,if x=-1 then it try to as many operations as possible
#now we can change algorithms from LinearRegression to svm or to clustering like clf=svm.SVR(kernel='poly')
clf.fit(X_train,y_train) #it will train our LinearRegression model and will give a hypothesis
with open ('LinearRegression.pickle','wb')as f:
    pickle.dump(clf,f)
'''
#f is a temporary variable,and we are dumping our trained classifier in 'f'
pickle_in=open('LinearRegression.pickle','rb') #opening the file LinearRegression.pickle in reading mode
clf=pickle.load(pickle_in) #again storing contents in clf classifier
accuracy=clf.score(X_test,y_test)
#print(X_test)?? why is it printing negative values??????
#print(accuracy)
#the model/ hypothesis that came ,now we are testing it with our test dataset
#this accuracy is actually squared error and it came 97% accurate
#print(X_lately)

forecast_set=clf.predict(X_lately)
#print(forecast_set,accuracy,forecast_out) #predicting and printing our label values of last 35 days(which initially have NA in their label values) ie from 2018-02-06
#to 27 march 2018
#print(df.head())
#print(df.tail())
df['Forecast']=np.nan       #column name Forecast is added to the dataFrame with Nan values till 5th feb 2018
#print(df.head())
#print(df.head())
#print(df.tail())

last_date=df.iloc[-1].name
# print(last_date) >>> prints 2018-02-05 00:00:00
'''
# Single selections using iloc and DataFrame
# Rows:
data.iloc[0] # first row of data frame (Aleshia Tomkiewicz) - Note a Series data type output.
data.iloc[1] # second row of data frame (Evan Zigomalas)
data.iloc[-1] # last row of data frame (Mi Richan)
# Columns:
data.iloc[:,0] # first column of data frame (first_name)
data.iloc[:,1] # second column of data frame (last_name)
data.iloc[:,-1] # last column of data frame (id)
'''

last_unix=last_date.timestamp()
#print(last_unix)
one_day=86400
next_unix=last_unix+one_day
#print(next_unix)

#all this heck under is done to have date in x-axis and also as our data set was cropped to 5th feb, it added dates and corresponding forecast values!!!
for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i] #[i] is value at index i

print(df.tail(36))
'''
note index of column start from 0 hence we are looping till df.colums-1
print(df.tail(36))

           Adj. Close    HL_PCT  PCT_change  Adj. Volume    label     Forecast
Date
2018-02-05     1068.76  4.325574    -2.89385    3742469.0  1006.94          NaN
2018-02-06         NaN       NaN         NaN          NaN      NaN  1106.429596
2018-02-07         NaN       NaN         NaN          NaN      NaN  1075.627956
2018-02-08         NaN       NaN         NaN          NaN      NaN  1023.940313
2018-02-09         NaN       NaN         NaN          NaN      NaN  1064.005086
2018-02-10         NaN       NaN         NaN          NaN      NaN  1075.543540
2018-02-11         NaN       NaN         NaN          NaN      NaN  1075.797206
2018-02-12         NaN       NaN         NaN          NaN      NaN  1094.718920
2018-02-13         NaN       NaN         NaN          NaN      NaN  1113.520240
2018-02-14         NaN       NaN         NaN          NaN      NaN  1117.334826
2018-02-15         NaN       NaN         NaN          NaN      NaN  1125.789766
2018-02-16         NaN       NaN         NaN          NaN      NaN  1135.616922
2018-02-17         NaN       NaN         NaN          NaN      NaN  1131.912743
2018-02-18         NaN       NaN         NaN          NaN      NaN  1150.973140
2018-02-19         NaN       NaN         NaN          NaN      NaN  1166.708485
2018-02-20         NaN       NaN         NaN          NaN      NaN  1138.847339
2018-02-21         NaN       NaN         NaN          NaN      NaN  1125.133769
2018-02-22         NaN       NaN         NaN          NaN      NaN  1091.425161
2018-02-23         NaN       NaN         NaN          NaN      NaN  1106.226267
2018-02-24         NaN       NaN         NaN          NaN      NaN  1117.167862
2018-02-25         NaN       NaN         NaN          NaN      NaN  1123.244970
2018-02-26         NaN       NaN         NaN          NaN      NaN  1137.842387
2018-02-27         NaN       NaN         NaN          NaN      NaN  1152.185240
2018-02-28         NaN       NaN         NaN          NaN      NaN  1183.964338
2018-03-01         NaN       NaN         NaN          NaN      NaN  1188.542755
2018-03-02         NaN       NaN         NaN          NaN      NaN  1161.200394
2018-03-03         NaN       NaN         NaN          NaN      NaN  1171.396763
2018-03-04         NaN       NaN         NaN          NaN      NaN  1173.250663
2018-03-05         NaN       NaN         NaN          NaN      NaN  1155.912526
2018-03-06         NaN       NaN         NaN          NaN      NaN  1121.077806
2018-03-07         NaN       NaN         NaN          NaN      NaN  1117.336195
2018-03-08         NaN       NaN         NaN          NaN      NaN  1115.757719
2018-03-09         NaN       NaN         NaN          NaN      NaN  1073.037183
2018-03-10         NaN       NaN         NaN          NaN      NaN  1046.310894
2018-03-11         NaN       NaN         NaN          NaN      NaN  1075.097467
2018-03-12         NaN       NaN         NaN          NaN      NaN  1025.326459
'''


df['Adj. Close'].plot() #first plot the Adj. Close(red) and then plot our forecast(blue)
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
