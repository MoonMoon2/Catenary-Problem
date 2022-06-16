import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from numpy import pi, sin

u0 = 0
uf = 0
g = 5


t_min = 0       # start time (seconds)
t_max = 5       # end time (seconds)
t_parts = 300    # how many partitions of time there are
nt = t_parts - 1    # for array iteration
dt = (t_max-t_min)/t_parts      # delta t
tpf = 1000*(t_max-t_min)/t_parts    # time per frame

x_min = 0       # starting point of chain (should be 0)
x_max = 50     # Length of hanging chain (meters)
x_parts = 200    # how many partitions should there be in the x direction
nx = x_parts - 1    # for array iteration
dx = (x_max-x_min)/x_parts  # delta x

v_initial = 0        # initial velocity

def vi(x):         # initial velocities
    return v_initial    # constant
    # return sin(x)

def f(x):           # initial positions
    #return 1/2 * x
    #return (x-25)**2
    #return (1/25)*(x-25)**2-25
    #return sin(2*x)
    x = sin(2*x)
    for j in range (0, len(x)-15):
        x[j] = 0
    return x


x = np.linspace(x_min, x_max, x_parts)
t = np.linspace(t_min, t_max, t_parts)

u = np.zeros((t_parts, x_parts))       # create the box for the solutions of x
u[0, :] = f(x)
u[:, 0] = u0
u[:, nx] = uf

for j in range(1, nx):
    c_1 = (g * dt ** 2) / (dx ** 2) + (g * dt ** 2) / (2 * dx)
    c_2 = 2 - (2 * g * dt ** 2) / (dx ** 2)
    c_3 = (g * dt ** 2) / (dx ** 2) - (g * dt ** 2) / (2 * dx)
    u[1, j] = c_1 * u[0, j+1] + c_2 * u[0, j] + c_1 * u[0, j-1] - dt * vi(x[j])

for i in range(1, nt):
    for j in range(1, nx):
        c_1 = (g * dt ** 2)/(dx ** 2) + (g * dt ** 2)/(2 * dx)
        c_2 = 2-(2 * g * dt ** 2)/(dx ** 2)
        c_3 = (g * dt ** 2)/(dx ** 2) - (g * dt ** 2)/(2 * dx)
        u[i+1, j] = c_1 * u[i, j+1] + c_2 * u[i, j] + c_3 * u[i, j-1] - u[i-1, j]

    print(u[i, :])

print(u[nt, :])

fig = plt.figure()
#ax1 = plt.subplot(111, xlim=(x.min(), x.max()), ylim=(u[1,:].min(), u[1,:].max()))
ax1 = plt.subplot(111, xlim=(x.min(), x.max()), ylim=(u.min(), u.max()))
ax1.title.set_text("Testing")

line1, = ax1.plot([],[],label="line")

ax1.legend()

def animate(i):
    line1.set_data(x,u[i,:])
    #dir = max(max(np.abs(u[i,:].min()), np.abs(u[i,:].max())), 1e-10)
    #ax1.set_ylim(-dir, dir)
    return line1

def init():
    line1.set_data([],[])
    return line1


ani = anim.FuncAnimation(fig, animate, np.arange(0,len(u[0,:])), init_func=init, interval=tpf)

plt.show()
