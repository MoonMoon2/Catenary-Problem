from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import sin, cosh, e

a = 5


def f(x, y):
    return (a/2)*cosh(x/a)+sin(y)


x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()

ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# intiates the viewbox 60 degrees up and 35 rotated
ax.view_init(25, 90)

plt.show()












