import math
import matplotlib.pyplot as plt

# pi/2 = 90˚
print('-------- angle in real num ---------')
print('sin(π/6) = %.3f' % math.sin(math.pi/6))
print('sin(π/3) = %.3f' % math.sin(math.pi/3))
print('sin(π/2) = %.3f' % math.sin(math.pi/2))
print('sin(π) = %.3f' % math.sin(math.pi))


print('-------- angle in ˚ ---------')
print('sin(30˚) = %.3f' % math.sin(math.radians(30)))
print('sin(60˚) = %.3f' % math.sin(math.radians(60)))
print('sin(90˚) = %.3f' % math.sin(math.radians(90)))
print('sin(180˚) = %.3f' % math.sin(math.radians(180)))


a = input("Print enter to see y = sin(X) in plot.... ")

X = [x/100 for x in range(1,1001)]
y = [math.sin(x) for x in X]

plt.plot(X,y)
plt.xlabel('X')
plt.ylabel('y = sin(X)')
plt.title('y = sin(X) in serias')
plt.show()
