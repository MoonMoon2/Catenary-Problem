import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from numpy import pi, sin, sqrt


u0 = 0
uf = -10
g = 11

def f(x):           # initial positions
    return 1/2 * x
    # return (x-25)**2
    # return (1/25)*(x-25)**2-25
    #return sin(4*pi*x/x_max)
    #return ((1/(x_max/uf))*x)*(1+sin(3*pi*x/x_max))
    # x = -(1/5)*x
    # for j in range(0, 25):
    #     x[j] = x[j]+(sin(3*pi*x[j]/x[25]))
    # return x



t_min = 0       # start time (seconds)
t_max = 20       # end time (seconds)
t_parts = 1000    # how many partitions of time there are

x_min = 0       # starting point of chain (should be 0)
x_max = 50     # Length of hanging chain (meters)
x_parts = int(((x_max-x_min)*t_parts)/(sqrt(g)*(t_max-t_min)))    # how many partitions should there be in the x direction
nx = x_parts - 1    # for array iteration
dx = (x_max-x_min)/x_parts  # delta x



nt = t_parts - 1    # for array iteration
dt = (t_max-t_min)/t_parts      # delta t
tpf = 1000*(t_max-t_min)/t_parts    # time per frame
v_initial = 0        # initial velocity

print("dt: {}, dxdt: {}".format(dt, dx))

def vi(x):         # initial velocities
    return v_initial    # constant
    # return sin(x)


x = np.linspace(x_min, x_max, x_parts)
t = np.linspace(t_min, t_max, t_parts)

u = np.zeros((t_parts, x_parts))       # create the box for the solutions of x
u[0, :] = f(x)
u[:, 0] = u0
u[:, nx] = uf

for j in range(1, nx):
    c_1 = (g * dt ** 2)/(2 * dx ** 2)
    c_2 = 1-(g * dt ** 2)/(dx ** 2)
    u[1, j] = c_1 * u[0, j+1] + c_2 * u[0, j] + c_1 * u[0, j-1] - dt * vi(x[j])

for i in range(1, nt):
    for j in range(1, nx):
        c_1 = (g * dt ** 2)/(dx ** 2)
        c_2 = 2-(2 * g * dt ** 2)/(dx ** 2)
        u[i+1, j] = c_1 * u[i, j+1] + c_2 * u[i, j] + c_1 * u[i, j-1] - u[i-1, j]


fig = plt.figure()
#ax1 = plt.subplot(111, xlim=(x.min(), x.max()), ylim=(u[1,:].min(), u[1,:].max()))
ax1 = plt.subplot(111, xlim=(x.min(), x.max()), ylim=(u.min(), u.max()))
ax1.title.set_text("")

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


ani = anim.FuncAnimation(fig, animate, np.arange(0,len(u[:,0])), init_func=init, interval=tpf)


plt.show()


# f = r"/Users/wardt/Desktop/3sinwaves5g.mp4"


f = r"/Users/wardt/Documents/PycharmProjects/Research/Output/simplewavefinitediffdirichlet-quadratic.gif"

# writervideo = anim.FFMpegWriter(fps=(1000/tpf))
# ani.save(f, writer=writervideo)

# writergif = anim.PillowWriter(fps=30)
# ani.save(f, writer=writergif)
