# .......................................Now we will write our own Linear Regression algorithm from scratch.......................................................

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random


style.use('fivethirtyeight') # Don't know what the hell it do??????

#xs=np.array([1,2,3,4,5,6],dtype=np.float64)
#ys=np.array([5,4,6,5,6,7],dtype=np.float64)

#creating random dataset
def create_dataset(hm,variance,step=2,correlation=False):
    val=1
    ys=[]
    for i in range(hm):                                                 #loop iterating hm times starting from zero
        y=val+random.randrange(-variance,variance)                      #maybe selecting a value between +- variance
        ys.append(y)                                                    #storing that value in ys
        if correlation and correlation=='pos':
            val+=step
        elif correlation and correlation=='neg':
            val-= step
    xs=[i for i in range(len(ys))]                      #now ys will be a list and hence for i starting from zero to len(ys)-1 ,is stored in xs
    return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
    m=( ((mean(xs)*mean(ys))-mean(xs*ys)) /
        ((mean(xs)*mean(xs))-mean(xs*xs)) )
    b=mean(ys)-m*mean(xs)
    return m,b

def squared_error(ys_orig,ys_line):
    return sum((ys_line-ys_orig)**2)

def coefficient_of_determination(ys_orig,ys_line):                              # coefficient_of_determination= r^2=1-(SE(regr)-SE(y mean))
    y_mean_line=[mean(ys_orig) for y in ys]                                     #it is a value and [] are for loop
    squared_error_regr=squared_error(ys_orig,ys_line)
    squared_error_y_mean=squared_error(ys_orig,y_mean_line)
    return 1-(squared_error_regr/squared_error_y_mean)

xs,ys=create_dataset(40,100,2,correlation='pos')
m,b=best_fit_slope_and_intercept(xs,ys)
#print(m,b)

Regression_line=[(m*x+b) for x in xs] #for loop to calculate mx+b for each x in xs, and hence it is basically a list

predict_x=8
predict_y=m*predict_x+b

r_squared=coefficient_of_determination(ys,Regression_line)
print(r_squared)

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,color='r')
plt.plot(xs,Regression_line)
plt.show()

'''
plt.scatter(x,y) #plt.plot(x,y)
plt.show()
'''
#......................................................Now we want to see ,how good is our best fit line..............................................................................
