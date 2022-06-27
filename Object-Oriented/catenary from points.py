from __future__ import annotations
import numpy as np
from numpy import arctan2, sin, cos, sqrt
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.integrate as integrate
from scipy.misc import derivative as dv


class Point:
    '''
    A class meant to represent a point in space with two neighbors, one to the left and one to the right

    ...

    Parameters
    ----------
    dx : float
        The current velocity in the x direction (meters per frame)
    dy : float
        The current velocity in the y direction (meters per frame)
    x, y : float
        The current x and y position of the point in space
    left, right : Point
        The neighboring point to the left and to the right of the current point

    Methods
    -------

    '''

    def __init__(self, x: float, y: float, is_endpoint : bool):
        '''
        Parameters
        ----------
        :param x: float x position of the point (meters)
        :param y: y position of the point (meters)
        :param is_endpoint : declares whether the point is an endpoint
        '''
        self.is_endpoint = is_endpoint

        self.x = x
        self.y = y
        # set temporary values for the constants, to be updated later
        self.dx = self.dy = 0.0
        self.mass = 0.0
        self.left = self.right = self
        self.left_initial_distance = self.right_initial_distance = 0
        self.net_force_x = self.net_force_y = 0
        self.left_distance = self.right_distance = self.length_of_segment = 0

    def __str__(self):
        return "({}, {}) is {}".format(self.x, self.y, "an endpoint" if self.is_endpoint else "not an endpoint")


    def set_neighbors(self, left_neighbor : Point, right_neighbor : Point):

        self.left = left_neighbor
        self.right = right_neighbor

        self.left_initial_distance = sqrt((self.x - self.left.x) ** 2 + (self.y - self.left.y) ** 2)/2
        self.right_initial_distance = sqrt((self.x - self.right.x) ** 2 + (self.y - self.right.y) ** 2)/2

        self.mass = linear_density * (self.left_initial_distance + self.right_initial_distance)


    def update(self):

        if (self.is_endpoint):
            return
        # update the distance to the left and right neighbors
        self.left_distance = sqrt((self.x - self.left.x) ** 2 + (self.y - self.left.y) ** 2)*0.5 #- (1-percentage_stretched_at_start)*self.left_initial_distance
        self.right_distance = sqrt((self.x - self.right.x) ** 2 + (self.y - self.right.y) ** 2)*0.5 #- (1-percentage_stretched_at_start)*self.right_initial_distance

        # find the angles to the left and right points from the current one. Uses the unit circle for direction
        a_left = arctan2(self.left.y - self.y, self.left.x - self.x)
        a_right = arctan2(self.right.y - self.y, self.right.x - self.x)

        self.length_of_segment = (self.left_distance / 2 + self.right_distance / 2)
        length_of_segment_x = self.left_distance * cos(a_left) + self.right_distance * cos(a_right)
        length_of_segment_y = self.left_distance * sin(a_left) + self.right_distance * sin(a_right)
        # update the forces towards the left and right neighbors based on hook's law
        # F = k * displacement
        f_left = s_c_corrected * self.left_distance
        f_right = s_c_corrected * self.right_distance

        # Forces to the left are negative and to the right are positive. This will be handled by the sin and cos
        # force of drag in the x direction
        drag_force_x = np.sign(self.dx) * drag_coefficient * fluid_density * length_of_segment_x * self.dx ** 2
        # $\cos{θ} = \frac{a}{h} = \frac{f_x}{f_s}$
        self.net_force_x = f_right * cos(a_right) + f_left * cos(a_left) - drag_force_x
        # f_grav = μ * length * grav_accel
        gravitational_force = self.mass * g_corrected
        # drag force in the y direction
        drag_force_y = np.sign(self.dy) * drag_coefficient * fluid_density * length_of_segment_y * self.dy ** 2
        # $\sin{θ} = \frac{a}{h} = \frac{f_y}{f_s}$
        self.net_force_y = f_left * sin(a_left) + f_right * sin(a_right) + gravitational_force - drag_force_y

        # acceleration = force/mass
        x_accel = self.net_force_x/self.mass
        y_accel = self.net_force_y/self.mass

        # update velocities and positions
        self.dx += x_accel
        self.x += self.dx
        self.dy += y_accel
        self.y += self.dy

        # print("Left Tension is {} and right is {}".format(f_left, f_right))




'''
Parameters
----------
:param spring_constant: float : the spring constant of the string being simulated. Controls how firmly the string is pulled together
'''
spring_constant = 805     # units: Newtons per meter
gravitational_acceleration = -9.81   # units: meters per second squared (negative is downwards)
linear_density = 5      # units: kilograms per meter
drag_coefficient = 100.3    # unitless, depends on shape (1.0-1.3 for rope https://www.engineeringtoolbox.com/drag-coefficient-d_627.html)
fluid_density = 1.225       # units: kilograms per meter cubed (1.225 for air)
percentage_stretched_at_start = 0

t0 = 0  # initial time (seconds)
tf = 30  # final time (seconds)
dt = 0.001  # size of a single partition of time
t = np.arange(t0, tf, dt)

# correcting constants for the time step size
# spring constant = N/m = Kg/s^2
# Needs to be multiplied by s^2/frame^2 which is dt^2
s_c_corrected = spring_constant * dt ** 2
# grav constant = m/s^2 which needs the same correction as above
g_corrected = gravitational_acceleration * dt ** 2


# Initial position as a function of x
# I will be setting the endpoints of my line based on this, so there is no need to worry.
# Create a function with the desired endpoints over a certain domain of x.
x0 = 0  # initial x bound in meters
xf = 10  # final x bound in meters
num_points = 50


def f(x):  # function for initial conditions
    return (1 / 5) * (x - 6) ** 2 + 1
    # a = linear_density / 0.0181
    # return a * np.cosh((x-5)/a) - 71.5097

# calculate the arc-length of the entire setup
def f_length_function(x):
    return sqrt(1 + dv(f, x, n=1) ** 2)


def f_length():  # returns the length of the curve f(x)
    length = 0.0
    x_line = np.linspace(x0, xf, 1000)
    for i in range(0, len(x_line) - 1):
        a = x_line[i]
        b = x_line[i + 1]
        length += integrate.quad(f_length_function, a, b)[0]

    print("Length: {}".format(length))

    return length

x, y = np.linspace(x0, xf, num_points), f(np.linspace(x0, xf, num_points))


points = []
for i in range(0, len(x)):
    points.append(Point(x[i], y[i], (i == 0 or i == len(x)-1)))

for i in range(0, len(x)):
    if points[i].is_endpoint:
        pass
    else:
        points[i].set_neighbors(points[i - 1], points[i + 1])

u = np.zeros((len(t), len(x)))
max_u = max(f(x))
min_u = min(f(x))
final_t_0 = 0
total_mass_at_end = 0
for i in range(0, len(t)):
    print("Frame #{}".format(i+1))
    for j in range(0, len(x)):
        u[i, j] = points[j].y
        points[j].update()

    for j in range(0, len(x)):
        if max_u < points[j].y:
            max_u = points[j].y
        elif min_u > points[j].y:
            min_u = points[j].y

        min_run = 0
        if i == len(t)-1:
            total_mass_at_end += points[j].mass
        if i == len(t)-1 and min_run > points[j].y:
            min_run = points[j].y
            final_t_0 = sqrt(points[j].net_force_x**2 + points[j].net_force_y**2)


        if points[j].is_endpoint:
            pass
        else:
            points[j].set_neighbors(points[j - 1], points[j + 1])


y = []
for i in range(0, len(x)):
    y.append(
        final_t_0/total_mass_at_end * np.cosh(x[i] * total_mass_at_end/final_t_0)
    )

print("{},{}".format(final_t_0, total_mass_at_end))

fig = plt.figure()
ax1 = plt.subplot(111, xlim=(min(x), max(x)), ylim=(min_u-1, max_u+1), xlabel="X (meters)", ylabel="Y (meters)")
# ax1 = plt.subplot(111, xlim=(x0, xf), ylim=(0, 10), xlabel="X (meters)", ylabel="Y (meters)")
ax1.title.set_text("Hanging chain made of small point masses")
ax1.legend()

line1, = ax1.plot([],[],label="points")
line2, = ax1.plot(x, u[len(t)-1, :],label="final")
line3, = ax1.plot(x, y, label="catenary")


def animate(i):
    line1.set_data(x, u[i, :])
    line2.set_data(x, u[len(t)-1, :])
    line3.set_data(x, y)
    pass

def init():
    line1.set_data([], [])
    line2.set_data(x, u[len(t) - 1, :])
    line3.set_data(x, y)
    pass


ani = anim.FuncAnimation(fig, animate, len(t), init_func=init, interval=(1000 * dt))

# f = r"/Users/wardt/Desktop/3sinwaves5g.mp4"
# writervideo = anim.FFMpegWriter(fps=(1000/tpf))
# ani.save(f, writer=writervideo)

# f = r"/Users/wardt/Documents/PycharmProjects/Research/Output/testing.gif"
# writergif = anim.PillowWriter(fps=1)
# ani.save(f, writer=writergif)

plt.show()
