import numpy as np

x={-1:np.array([[1,7],
                [2,8],
                [3,8]]),

    1:np.array([[5,1],
                [6,-1],
                [7,3]])}

all_data=[]
dic={}
'''
for yi in x:
    for featureset in x[yi]:
        for feature in featureset:
            all_data.append(feature)

print(all_data)
'''
b=11
w1=[4,3]
w2=[4,-3]
w3=[-4,3]
w4=[12,5]
dic[5]=[w1,b]
dic[5]=[w2,b]
dic[5]=[w3,b]
dic[13]=[w4,b]

n=sorted([n for n in dic]) # it will sort the dictionary in order of magnitudde  of w and store them in norms
print(n)
print(20*'#')
opt_choice=dic[n[0]]
print(dic[n[0]])
print(20*'#')
w=opt_choice[0]
print(w)
print(20*'#')                # opt choice can have many w's with same magnitude eg [5,5] and [5,-5]
b1=opt_choice[1]
print(b1)
print(20*'#')
print(dic)


'''
Output
[5, 13]
####################
[[-4, 3], 11]
####################
[-4, 3]
####################
11
####################
{5: [[-4, 3], 11], 13: [[12, 5], 11]}
'''
'''
How will you know that you ought to take a next step?
Ans-We know for support vectors the value of equation yi(xi.w+b)=1 hence if in both classes we found some vectors for which the value of this equation is very close to 1 (which depends on you how close you want), we will say we have find the optimized value of w and b, if its not the case we will take next step!!!
'''
