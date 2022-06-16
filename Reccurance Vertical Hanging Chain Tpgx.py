import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from numpy import sin


'''
t: uses i for iteration, is the time variable
x: uses j for iteration, is the space variable
'''
# ------------------------------------------------------
#            Boundary Conditions and Constants
# ------------------------------------------------------

u0 = 1e-15          # x=0 is 0 at all t
uf = 1          # Final boundary
g = -9.80665    # Gravitational Constant (m/s^-2)

x_min = 0       # starting point of chain (should be 0)
x_max = 2     # Length of hanging chain (meters)
x_parts = 500    # how many partitions should there be in the x direction
nx = x_parts - 1    # for array iteration

dx = (x_max-x_min)/x_parts  # delta x


t_min = 0       # start time (seconds)
t_max = 1       # end time (seconds)
t_parts = 500    # how many partitions of time there are
nt = t_parts - 1    # for array iteration
dt = (t_max-t_min)/t_parts
tpf = 1000*(t_max-t_min)/t_parts
initial_velocity_const = 0


def v_i(x):         # initial velocities
    return initial_velocity_const +x-x    # constant
    # return sin(x)


def f(x):           # initial positions
    return 1/2 * x

# ------------------------------------------------------
#               Solving the PDE
# ------------------------------------------------------

x = np.linspace(x_min, x_max, x_parts)

t = np.linspace(t_min, t_max, t_parts)

u = np.zeros((t_parts, x_parts))       # create the box for the solutions of x
u[0, :] = f(x)
u[:, 0] = u0
u[:, nx] = uf

vel_i = v_i(x)          # initial velocity (for phantom points)

print(vel_i)
# populate u[1, :] using phantom points (velocity)
for j in range(1, nx-1):
    c_1 = ((2 * x[j] + dx) * g * dt ** 2) / (2 * dx ** 2)
    c_2 = 2 - ((2 * x[j] * g * dt ** 2) / (dx ** 2))
    c_3 = ((2 * x[j] - dx) * g * dt ** 2) / (2 * dx ** 2)
    u[1, j] = (c_1 * u[0, j + 1] + c_2 * u[0, j] + c_3 * u[0, j - 1]) / 2 - dt * vel_i[j]
print("Init2")
# populate u based on initial conditions and previous values
# leaving out u[:, x_parts-1] because
for i in range(0, nt-1):
    print("%3.1f" % ((i/nt)*100))
    for j in range(1, nx-1):
        c_1 = ((2 * x[j] + dx) * g * dt ** 2) / (2 * dx ** 2)
        c_2 = 2 - ((2 * x[j] * g * dt ** 2) / (dx ** 2))
        c_3 = ((2 * x[j] - dx) * g * dt ** 2) / (2 * dx ** 2)
        u[i + 1, j] = c_1 * u[i, j + 1] + c_2 * u[i, j] + c_3 * u[i, j - 1] - u[i - 1, j]

print(u)


fig = plt.figure()
ax1 = plt.subplot(111, xlim=(x.min(), x.max()), ylim=(u[1,:].min(), u[1,:].max()))
#ax1 = plt.subplot(111, xlim=(x.min(), x.max()), ylim=(u.min(), u.max()))
ax1.title.set_text("Testing")

line1, = ax1.plot([],[],label="bruh")

ax1.legend()

def animate(i):
    line1.set_data(x,u[i,:])
    dir = max(np.abs(u[i,:].min()), np.abs(u[i,:].max()))
    ax1.set_ylim(-dir, dir)
    return line1
def init():
    line1.set_data([],[])
    return line1


ani = anim.FuncAnimation(fig, animate, np.arange(0,len(t)), init_func=init, interval=tpf)

plt.show()