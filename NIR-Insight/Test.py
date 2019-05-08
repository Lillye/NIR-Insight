from Local import *

for i in range(1,11):
    for j in range(i+1,11):
        print('Test :' + str(i) + ' ' + str(j))
        Local(i,j)
        print()