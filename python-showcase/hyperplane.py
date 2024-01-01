import math
import matplotlib.pyplot as plot




X = [x-1 for x in range(0,101)]
y = [-x for x in X] # y = -x => x + y = 0
y1 =[10-x for x in X] # y = -x + 1 => x + y = 10

#plot.plot(X,y)
#plot.plot(X,y1)
#plot.show()



X = [x-1 for x in range(0,10)]
y = [x * x for x in X] # y = x * x => x * x - y = 0
y1 =[x * x - 100 for x in X] # y = x * x - 100 => x * x - y  = 100
plot.plot(X,y)
plot.plot(X,y1)
plot.show()

# 等号右边的值，就类似等高线的刻度。
