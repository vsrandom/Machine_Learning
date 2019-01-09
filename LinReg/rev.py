import numpy as np
from matplotlib import pyplot as plt
'''
a=np.arange(16)
print(a)
print('\n')
print(a.ndim)
b=a.reshape(4,2,2)
print(b)
'''
'''
a=np.empty([3,2],dtype=float)
print(a)
'''
'''
a=np.zeros([2,2],dtype=int)
print(a)
print('\n')
b=np.ones([5,5],dtype=complex)
print(b)

https://stackoverflow.com/questions/626759/whats-the-difference-between-lists-and-tuples
list cannot be used as key in dictionary ,while tuple can be used!
'''
'''
b=[1,2,3]
print(b)  #simple list we can to operations on it!
a=[(1,2,3),(4,5)] #list of tuples
x=np.asarray(a)
print(x.ndim,x)
'''
'''
error
a="Hello world"
x=np.frombuffer(a,dtype='S1')
print(x)
'''
'''
a=range(5)
it=iter(a)
x=np.fromiter(it,dtype=float)
print(x)
'''
'''
a=np.arange(10,30,2) #start end and step
print(a)
b=a[0:9:3]  #starting from zero and not including the 9th index and having steps of two!
print(b)
'''

'''
a=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
b=a[0:3] #[0,3)
print(b)
'''
'''
a=np.array([[1,2,3],[4,5,6],[7,8,9]])
b=a[1:2,0:3]
c=a[[0],0:3]
d=a[[1],[1]]
e=a[1:3,1:3]
print(b,c,d,2*'\n',e)
'''

'''
a=[1,2,3]
b=np.array(a,dtype=float)
print(b)
'''
'''

a=np.arange(0,60,5)
print(a[...])
b=a.reshape(3,4)
print(b[0,1])      #0th row and first column !!
print(2*'\n')
print(b[...])      #printing b
for x in np.nditer(b,order='F'):
    print(x)

'''
'''
a=np.arange(10)
b=np.split(a,[0,2])  #make a partition before 0th index and 2nd index !! hence there will be three arrays!!
 #output [array([], dtype=int64), array([0, 1]), array([2, 3, 4, 5, 6, 7, 8, 9])] see empty array before 0 index of datatype int
print(b)
'''

'''
a=np.array([-1.5,-2.6,1.5])
b=np.ceil(a)
c=np.floor(a)
print(b,c)
'''

a=np.arange(1,11) #from [1,10)
y=2*a+5   #also an array !!
z=np.sin(a)
t=np.cos(a)
plt.xlabel("x axis")
plt.ylabel("y axis")

plt.subplot(3,1,1) #3 graphs and 1 width and 1st is active
plt.plot(a,y,'o',color='r')
plt.subplot(3,1,2)
plt.plot(a,z,color='k')
plt.subplot(3,1,3)
plt.plot(a,t)

#plt.title("Demo")
#plt.plot(a,y,'x',color='r')
plt.show()
