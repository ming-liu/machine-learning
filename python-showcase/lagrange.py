import math
import matplotlib.pyplot as plot


# [Math & Algorithm]拉格朗日乘数法

# x^2 + y^2 = 1
# y = 1 - x^2
# X = [x/10 for x in range(1,111)] + [-x/10 for x in range(1,111)]
X1 = [x/10 for x in range(1,11)]
y1 = [(1-x*x) ** 0.5 for x in X1]

X2 = [-x/10 for x in range(1,11)]
y2 = [(1-x*x) ** 0.5 for x in X2]

# plot.plot(X1,y1,'o',markersize=1)
plot.plot(X1,y1)
plot.plot(X2,y2)
plot.show()
