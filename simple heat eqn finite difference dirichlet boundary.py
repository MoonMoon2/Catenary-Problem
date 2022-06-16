import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from numpy import sin

# initial conditions
def f(x):
    return 2*x

c = -1

# boundarie conditions
u0 = 0
uf = 0

x_min = 0
x_max = 10
x_partitions = 25
nx = x_partitions-1

t_min = 0
t_max = 0.5
t_partitions = 500
nt = t_partitions-1




def f(x):
    return -(1/(x_max/2))*(x-(x_max/2))**2 + (x_max/2)
    #return -(1/2.5)*(x-2.5)**2 + 2.5
    #return sin(x)


# arrays
t = np.linspace(t_min, t_max, t_partitions)
x = np.linspace(x_min, x_max, x_partitions)

dt = t[1]-t[0]
dx = x[1]-x[0]

u = np.zeros((t_partitions, x_partitions))
u[:, 0] = u0
u[:, nx] = uf
u[0, :] = f(x)

print(len(u))
print(len(u[0,:]))

for i in range(0, nt):
    for j in range(1, nx):
        u[i+1, j] = ((c*dt)/(dx**2))*u[i, j+1] + (1+(2*c*dt)/(dx**2))*u[i,j] + ((c*dt)/(dx**2))*u[i, j-1]


fig = plt.figure()

#ax1 = plt.subplot(111, xlim=(x.min(), x.max()), ylim=(u[1,:].min(), u[1,:].max()))
ax1 = plt.subplot(111, xlim=(x.min(), x.max()), ylim=(u.min(), u.max()))
ax1.title.set_text("Animation")

line1, = ax1.plot([],[],label="diffusion")

line2, = ax1.plot([], [],label="initial")
ax1.legend()

def animate(i):
    line1.set_data(x,u[i,:])
    #dir = max(max(np.abs(u[i,:].min()), np.abs(u[i,:].max())), 1e-10)
    #ax1.set_ylim(-dir, dir)
    return line1

def init():
    line1.set_data([],[])
    line2.set_data(x, u[0, :])
    return line1


def tpf():
    #return 1000*((t_max-t_min)/t_partitions)
    return 20

ani = anim.FuncAnimation(fig, animate, np.arange(0,t_partitions), init_func=init, interval=tpf())

plt.show()