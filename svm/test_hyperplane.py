import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

ax=plt.axes()
#ax.plot([1,2],[1,2],'k')
w=[0.224, -0.224]
b=0.11999999999914479

def hyperplane(x,w,b,v):
    return (-w[0]*x-b+v)/w[1]

min_feature_value=-1
max_feature_value=8
datarange=(min_feature_value*0.9,max_feature_value*1.1)
hyp_x_min=datarange[0]
hyp_x_max=datarange[1]


psv1=hyperplane(hyp_x_min,w,b,1)
psv2=hyperplane(hyp_x_max,w,b,1)
ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2],'k')

nsv1=hyperplane(hyp_x_min,w,b,-1)
nsv2=hyperplane(hyp_x_max,w,b,-1)
ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2],'k')

dsv1=hyperplane(hyp_x_min,w,b,0)
dsv2=hyperplane(hyp_x_max,w,b,0)
ax.plot([hyp_x_min,hyp_x_max],[dsv1,dsv2],'y--')

plt.show()
