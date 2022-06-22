'''
Lagrangian Hanging Chain Simulation with Dirichlet Boundary Conditions
@author: T. Ward

----------------------------------------------------------------------

Using an tuple to represent a single line segment, each with an array inside.

TODO: Methods needed
    - Create a line segment
    - Find tension at a point
    - Add tension based on distance from endpoint to endpoint (force them together)
        - damping for the accel?
    
----------------------------------------------------------------------

----------------------------------------------------------------------

Variables:

Classes and their variables:

Segment:
    Represents a short line segment.

    Has the following member variables:
        - x: horizontal position of the center-point of the line segment
        - y: vertical position of the center-point of the line segment
        - dxdt: the horizontal velocity of the line segment
        - dxdtt: the horizontal acceleration of the line
        - dydt: the vertical velocity of the line segment
        - dydtt: the vertical acceleration of the line
        - a: angle at which the segment currently is in relation to the x axis
        - dadt: the angular velocity of the line segment due to torque
        - dadtt: the angular acceleration of the line segment due to torque
        - T1: magnitude of the tension acting on the left end of the line segment
        - T2: magnitude of the tension acting on the right end of the line segment

    Has the following methods:

'''
import numpy as np
from numpy import sin, cos, tan, e, sqrt, arctan, pi, arccos
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.integrate as integrate
from scipy.misc import derivative as dv

'''
Class of objects which hold and have all of our methods for controlling and updating a segment of our line
'''


class Segment:
    # Instance Variables
    dxdt = dxdtt = dydt = dydtt = dadt = dadtt = 0

    def __init__(self, x1, y1, x2, y2, is_start, is_end):
        self.pos1 = x1, y1
        self.pos2 = x2, y2

        self.iS = is_start
        self.iE = is_end

        if self.iS:
            self.x, self.y = self.pos1
        elif self.iE:
            self.x, self.y = self.pos2
        else:
            self.x, self.y = ((x1 + x2) / 2), ((y1 + y2) / 2)  # average position of the endpoints

        self.s = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # fix the length so it won't change
        self.T1 = self.T2 = ((y1 + y2) / 2) * sqrt(1 + ((y2 - y1) / (x2 - x1)) ** 2)
        self.a = arctan((y2 - y1) / (x2 - x1))

    def update(self, seg1, seg2):
        # Grab the important details of the segments and convert them to angles for our forces and torque
        # these angles are towards the correct endpoint, so the forces will hold the ends of the line segments together
        d_to_seg1 = sqrt((seg1.pos2[1] - self.pos1[1]) ** 2 + (seg1.pos2[0] - self.pos1[0]) ** 2)
        a1 = seg1.a if d_to_seg1 == 0 else \
            -(arccos((seg1.pos2[0] - self.pos1[0]) / d_to_seg1) if seg1.pos2[0] != self.pos1[0] else pi/2) if seg1.pos2[1] > self.pos1[1] else \
                arccos((seg1.pos2[0] - self.pos1[0]) / d_to_seg1) if seg1.pos2[0] != self.pos1[0] else pi/2


        d_to_seg2 = sqrt((seg2.pos1[1] - self.pos2[1]) ** 2 + (seg2.pos1[0] - self.pos2[0]) ** 2)
        a2 = seg2.a if d_to_seg2 == 0 else \
            -(arccos((seg2.pos1[0] - self.pos2[0]) / d_to_seg2) if seg2.pos1[0] != self.pos2[0] else pi/2) if seg2.pos1[1] < self.pos2[1] else \
                arccos((seg2.pos1[0] - self.pos2[0]) / d_to_seg2) if seg2.pos1[0] != self.pos2[0] else pi/2

        # set the angular acceleration for each type of segment
        if self.iS:  # This is the leftmost segment, which is attached on the left end
            self.dadtt = self.s * (self.T2 * sin(a2)) / ((1 / 3) * lden * self.s ** 3)
        elif self.iE:  # this is the rightmost segment, which is attached at the right end
            self.dadtt = self.s * (self.T1 * sin(a1)) / ((1 / 3) * lden * self.s ** 3)
        else:  # This is a regular segment, and needs to be updated from both ends
            self.dadtt = self.s * (self.T2 * sin(a2) + self.T1 * sin(a1)) / (
                    (1 / 3) * lden * self.s ** 3)

        # now lets update our angular velocity and angle appropriately, by multiplying each by the time step

        self.dadt = self.dadtt * dt
        self.a += self.dadt * dt

        # We must update the positions individually as well because they use the same code
        if self.iS:  # the left end is fixed, so we only need to update the right end
            self.pos2 = ((self.pos1[0] + self.s * cos(self.a)), (self.pos1[1] + (self.s) * sin(self.a)))
        elif self.iE:  # the right end is fixed, so we only need to update the left end
            self.pos1 = (self.x + (self.s) * cos(pi + self.a)), (self.y + (self.s) * sin(pi + self.a))
        else:  # neither end is fixed, so we must physically move the line beyond just rotation
            a2 += self.a
            a1 += self.a
            # The horizontal acceleration is given by
            # μ*s*dxdtt = T2 * cos(a2) - T1 * cos(a1)
            # solved for dxdtt, we have the following:
            self.dxdtt = (self.T2 * cos(a2) - self.T1 * cos(a1)) / (lden * self.s)

            # the vertical acceleration is the same, except taking the sin of the angles (because sin is
            # y component) and incorporating gravity into the equation (μ*g*s)
            self.dydtt = (self.T2 * sin(a2) - self.T1 * sin(a1) - lden * (g * dt ** 2) * self.s) / (lden * self.s)

            # for our velocity and position we will increment corrected for our frames per second

            self.dxdt = self.dxdtt * dt
            self.x += self.dxdt * dt
            self.dydt = self.dydtt * dt
            self.y += self.dydt * dt

            # update the endpoints based on the new centerpoint, and current rotation
            self.pos1 = (self.x + (self.s / 2) * cos(pi + self.a)), (self.y + (self.s / 2) * sin(pi + self.a))
            self.pos2 = ((self.x + (self.s / 2) * cos(self.a)), (self.y + (self.s / 2) * sin(self.a)))

        # now we set the magnitude of the tension for the next round
        # y * sqrt(1 + tan²(θ)) is the tension based on a preset curve
        # self.T1 = self.pos1[1] * sqrt(1 + (tan(self.a)) ** 2)
        # self.T2 = self.pos2[1] * sqrt(1 + (tan(self.a)) ** 2)
        # This version uses the distance between the endpoints to determine the tension.
        # DOES NOT ACCOUNT FOR ENDPOINTS BECAUSE THEY DON'T USE THE TENSION AS OF RIGHT NOW
        self.T1 = spring_constant * sqrt((seg1.pos2[1] - self.pos1[1]) ** 2 + (seg1.pos2[0] - self.pos1[0]) ** 2)
        self.T2 = spring_constant * sqrt((seg2.pos1[1] - self.pos2[1]) ** 2 + (seg2.pos1[0] - self.pos2[0]) ** 2)

        # self.T1 = self.pos1[1] * sqrt(1 + tan(self.a) ** 2) + spring_constant * sqrt(
        #     (seg1.pos2[1] - self.pos1[1]) ** 2 + (seg1.pos2[0] - self.pos1[0]) ** 2)
        # self.T2 = self.pos2[1] * sqrt(1 + tan(self.a) ** 2) + spring_constant * sqrt(
        #     (seg2.pos1[1] - self.pos2[1]) ** 2 + (seg2.pos1[0] - self.pos2[0]) ** 2)

        # Limit the lines to the bounding box TODO: temporary for this viewbox, remove or correct once catenary works
        self.pos1 = ((self.pos1, (10, self.pos1[1]))[self.pos1[0] > 10], (0, self.pos1[1]))[self.pos1[0] < 0]
        self.pos2 = ((self.pos2, (10, self.pos2[1]))[self.pos2[0] > 10], (0, self.pos2[1]))[self.pos2[0] < 0]
        self.pos1 = ((self.pos1, (self.pos1[0], 10))[self.pos1[1] > 10], (self.pos1[0], 0))[self.pos1[1] < 0]
        self.pos2 = ((self.pos2, (self.pos2[0], 10))[self.pos2[1] > 10], (self.pos2[0], 0))[self.pos2[1] < 0]

    def __str__(self):
        return "Line from {} to {}, at angle {}".format(self.pos1, self.pos2, self.a)

    def points(self):
        return self.pos1, self.pos2

    def x_values(self):
        return self.pos1[0], self.pos2[0]

    def y_values(self):
        return self.pos1[1], self.pos2[1]


'''
----------------------------------------------------------------------

Now begins the actual code for our situation. The following variables 
control the shape, weight, and conditions in which our hanging chain 
finds itself.

----------------------------------------------------------------------
'''
lden = 1  # linear density
g = 9.81  # acceleration due to gravity. (downwards force) Assumed positive, and subtracted in calculations
spring_constant = 50

t0 = 0  # initial time (seconds)
tf = 30  # final time (seconds)
dt = 0.01  # size of a single partition of time
t = np.arange(t0, tf, dt)
'''
Initial position as a function of x 
I will be setting the endpoints of my line based on this, so there is no need to worry.
Create a function with the desired endpoints over a certain domain of x.
'''
x0 = 0  # initial x bound in meters
xf = 10  # final x bound in meters


def f(x):  # function for initial conditions
    return (1 / 5) * (x - 5) ** 2 + 1


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


'''
length of a single segment - smaller is better, but too small leads to truncation error
The smallest this can be to avoid truncation error is 5.53e-103 for values of s,
but setting it below around 1e-15 is enough to cause major problems for your data
(ballooning due to dividing by such a small number)
'''
s = 1
'''
The next line sets up my x values for each line segment as an array, as well as the array of
corresponding initial y values.

num_segments = f_length()/s
'''
x, y = np.linspace(x0, xf, 4000), f(np.linspace(x0, xf, 4000))

# int((f_length()/s)//sqrt(2))
'''
Now we create the line segments, and store them in an array.
'''
line_segments = []
for i in range(0, len(x) - 1):
    line_segments.append(Segment(x[i], y[i], x[i + 1], y[i + 1], i == 0, i == (len(x) - 2)))
'''
Animation initialization
'''

'''
Here we create an array which holds each of the start and end points 
'''
animated_lines = []
'''
Animate
'''
for each in line_segments:
    animated_lines.append([])

x_min = x_max = x[0]
y_min = y_max = y[0]

for i in range(0, len(t)):
    # Save the current state of the lines in the current time slot of each of their respective animated line
    # Uses a index call to get the index for animated lines, and then uses the advanced for loop to make
    # the rest simple
    buffer_segments = []
    for segment in line_segments:
        animated_lines[line_segments.index(segment)].append((segment.x_values(), segment.y_values()))
        buffer_segments.append(segment)
        x_min = min(segment.x_values()) if min(segment.x_values()) < x_min else x_min
        x_max = max(segment.x_values()) if max(segment.x_values()) > x_max else x_max
        y_min = min(segment.y_values()) if min(segment.y_values()) < y_min else y_min
        y_max = max(segment.y_values()) if max(segment.y_values()) > y_max else y_max

    # now to update them: The first and last dont have an element to their left or right respectively,
    # so will be updated outside the for loop
    # NOTE: The update propagates from left to right, so any strange tendencies related to this propagation
    # will be hard to fix
    # TODO: might need to change update order from the outer edges in - also lock in the end lines dx and dy?


    line_segments[0].update(buffer_segments[0], buffer_segments[1])  # update first segment
    line_segments[-1].update(buffer_segments[-2], buffer_segments[-1])  # update last segment

    for j in range(1, len(line_segments)//2):
        line_segments[j].update(buffer_segments[j - 1], buffer_segments[j + 1])
        line_segments[-j-1].update(buffer_segments[-j - 2], buffer_segments[-j])

    if len(line_segments)%2 == 1:
        line_segments[len(line_segments)//2].update(buffer_segments[len(line_segments)//2-1], buffer_segments[len(line_segments)//2+1])

fig = plt.figure()
# ax1 = plt.subplot(111, xlim=(x_min, x_max), ylim=(y_min, y_max))
ax1 = plt.subplot(111, xlim=(x0, xf), ylim=(0, 10))
ax1.title.set_text("Animation of a hanging chain (still testing)")

displayed_segments = []
for i in range(0, len(animated_lines)):
    line, = ax1.plot([], [], label="Segment {}".format(i + 1))
    displayed_segments.append(line, )

for j in range(0, len(displayed_segments)):
    displayed_segments[j].set_data(animated_lines[j][0][0], animated_lines[j][0][1])


def animate(i):
    for j in range(0, len(displayed_segments)):
        displayed_segments[j].set_data(animated_lines[j][i][0], animated_lines[j][i][1])

    # print(animated_lines[4][i])
    return displayed_segments


def init():
    for line in displayed_segments:
        line.set_data([], [])
    return displayed_segments


ani = anim.FuncAnimation(fig, animate, len(t), init_func=init, interval=(1000 * dt))

# f = r"/Users/wardt/Desktop/3sinwaves5g.mp4"
# writervideo = anim.FFMpegWriter(fps=(1000/tpf))
# ani.save(f, writer=writervideo)

# f = r"/Users/wardt/Documents/PycharmProjects/Research/Output/lagrange-v2.gif"
# writergif = anim.PillowWriter(fps=1)
# ani.save(f, writer=writergif)

plt.show()
