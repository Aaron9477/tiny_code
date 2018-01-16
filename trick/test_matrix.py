import time
import math as m
import numpy as np

a = b = np.random.random((100,100,100,100))
# print(m.sin(a[0][0][0][0]))
print(a)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
start = time.clock()
a[0] += b[0]
print('time: ', time.clock()-start)
print(a)
