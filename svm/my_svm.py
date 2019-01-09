import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')


class Support_Vector_Machine:
    def __init__(self,visualization=True):  #sort of object and __init__ is a constructor and we are actually plotting our data using visualization
        self.visualization=visualization
        self.colors={1:'r',-1:'b'}
        if self.visualization:
            self.fig=plt.figure()
            self.ax=self.fig.add_subplot(1,1,1)  #here 1,1 means add grid lines and 3rd 1 is we are plotting only one graph!!


    def fit(self,data):
        self.data=data
        opt_dict={} # A dictionary to store data in the form {||w||:[w,b]} and note although we will test 4 cases for each w but we will also be replacing it and hence in the end for a corresponding ||w|| there will be one pair of w and b
        transforms=[[1,1],     #for each ||w|| there can be 4 cases
                    [-1,1],
                    [-1,-1],
                    [1,-1]]


        all_data=[]
        for yi in self.data:                   #selecting class ie 1 or -1
            for featureset in self.data[yi]:   #for each point of that class
                for feature in featureset:     #both x and y value
                    all_data.append(feature)   #all_data contain individual values now [1, 7, 2, 8, 3, 8, 5, 1, 6, -1, 7, 3]


        self.max_feature_value=max(all_data)   #self.max_feature_value=8
        self.min_feature_value=min(all_data)   #self.min_feature_value=-1
        all_data=None                          #clearing data

        step_sizes=[self.max_feature_value*0.1, #list of step sizes we want to take for w to decrease ie 0.8,0.08,0.008
                    self.max_feature_value*0.01,
                    #point_of expense
                    self.max_feature_value*0.001]

        #extremely expensive
        b_range_multiple=5  #now it is an expensive step hence we don't much care about precision of b
        b_multiple=5
        latest_optimum=self.max_feature_value*10 #latest_optimum=80

        '''
        Now we have to find min value of ||w|| and this comment will try to tell what is happening under this..
        -> now for each step in step_sizes ie first one is 0.8, we first initialize w=[x,x] and set the optimized value to be false
        Now until optimized is not true for each value in range of b
            and for each transformation for a particular w
            we are going to see the if the value of yi*(np.dot(w_t,xi)+b)>=1 is true or not for all data points of both class and if even a single value is prove  unable to prove this we will discard
            that w value.
            but in case that w value saatisfies that equation,we are going to store it in dictionary opt_dict, and note for each ||w|| there is going to be a single [w,b] as like if ist and 4th are the transformation which satisfies for each point then 4th will be the value as it will overwrite ist value!!
            (As per what i tested using sample code in test_svm.py)
        Now about w[0]<0?
        Ans-the initial problem was to minimize |w| and its a convex problem so minimum of |w|, w being a vector is 0 as we may or may never reach 0 so checking <0 i guess ... not sure if i am correct﻿
        **important**
        latest_optimum=opt_choice[0][0]+step*2
        what above line means is ,now w is coming down and if it crossed the bottom implie it has to go backward and hence there is your step*2 for x coordinate of w, now on performing this step we are sure we are before the optimum walue of w,now further we cn take smaller steps in forward direction,
            Now it will happen for 0.08 and 0.008 as well and hence after that we will get our self.w and self.b and hence we trained out classifier or we completed our 'fit' method.
        (As far i understood!!)
        '''
        for step in step_sizes:
            w=np.array([latest_optimum,latest_optimum])
            optimized=False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),(self.max_feature_value*b_range_multiple),step*b_multiple):
                    for transformation in transforms:
                        w_t=w*transformation
                        found_option=True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b)>=1:
                                    found_option=False

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)]=[w_t,b]

                if w[0]<0:
                    optimized=True
                    print('optimized a step')
                else:
                    w=w-step

            norms=sorted([n for n in opt_dict]) # it will sort the dictionary in order of magnitudde  of w and store them in norms
            opt_choice=opt_dict[norms[0]]       #eg opt_choice=[[-4, 3], 11]
            self.w=opt_choice[0]
            self.b=opt_choice[1]
            latest_optimum=opt_choice[0][0]+step*2

        for i in self.data: #printing value of yi*(np.dot(self.w,xi)+self.b for each training example.
            for xi in self.data[i]:
                yi=i
                print(xi,':',yi*(np.dot(self.w,xi)+self.b))



    def predict(self,features):
        classification=np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification!=0 and self.visualization:
            self.ax.scatter(features[0],features[1],s=200,marker='*',c=self.colors[classification]) # self.colors={1:'r',-1:'b'}
            #now for each feature we are plotting it with * and color will tell if it belongs to '1' class or '-1' class
        return classification
    #...............................................................................................................................................
    #Now we have completed our training and testing method, now its time for plotting support vector hyperplanes and decision boundary..

    def visualize(self): #plotting each data point of data_dict
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        #hyperplane=x.w+b
        #v=x.w+b
        #psv=1
        #nsv=-1
        #dec=0
        '''
        Now run this code and you will know what ax.plot is doing

        import matplotlib.pyplot as plt
        from matplotlib import style
        style.use('ggplot')

        ax=plt.axes()
        ax.plot([1,2],[1,2],'k')

        plt.show()

        ie
            ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2],'k')
            this line means that ax will create a straight line joining (hyp_x_min,psv1) and (hyp_x_max,psv2) ,starting from (hyp_x_min,psv1) and
            ending at (hyp_x_max,psv2) and yeah 'k' means color will be black!! 
        '''
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v)/w[1]

        datarange=(self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min=datarange[0]
        hyp_x_max=datarange[1]

        #w.x+b=1
        #positive support vector hyperplane
        psv1=hyperplane(hyp_x_min,self.w,self.b,1)
        psv2=hyperplane(hyp_x_max,self.w,self.b,1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2],'k')

        #(w.x+b)=-1
        #negative support vector hyperplane
        nsv1=hyperplane(hyp_x_min,self.w,self.b,-1)
        nsv2=hyperplane(hyp_x_max,self.w,self.b,-1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2],'k')

        #(w.x+b)=0
        #negative support vector hyperplane
        dsv1=hyperplane(hyp_x_min,self.w,self.b,0)
        dsv2=hyperplane(hyp_x_max,self.w,self.b,0)
        self.ax.plot([hyp_x_min,hyp_x_max],[dsv1,dsv2],'y--')

        print(20*'#')
        print(self.w,self.b)

        plt.show()




data_dict={-1:np.array([[1,7],
                        [2,8],
                        [3,8]]),

            1:np.array([[5,1],
                        [6,-1],
                        [7,3]])}

svm=Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us=[[0,10],
            [1,3],
            [3,4],
            [3,5],
            [5,5],
            [5,6],
            [6,-5],
            [5,8]]

for p in predict_us:
    svm.predict(p)

svm.visualize()
