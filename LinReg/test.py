import numpy as np

x={2:[[1,2,3,4,5,6],[23,43,3,12,45,43]], 4:[[21,2,3,4,1,34],[28,34.66,90,100,45]]}
print(x)
print(20*'#')
for group in x:
    for data in x[group]:
        print(data)
