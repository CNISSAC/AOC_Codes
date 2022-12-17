import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Example simulation of two-link arm dynamics using discrete dynamics

class Arm:

  def __init__(self):

    # model parameters
    self.m1 = 1                     # mass of first body
    self.m2 = 1                     # mass of second body
    self.l1 = 0.5                   # length of first body
    self.l2 = 0.5                   # length of second body
    self.lc1 = 0.25                 # distance to COM
    self.lc2 = 0.25                 # distance to COM
    self.I1 = self.m1*self.l1/12.0  # inertia 1
    self.I2 = self.m2*self.l2/12.0  # inertia 2
    self.g = 9.8                    # gravity
    self.tf = 2.0                   # final time
    self.N = 128                    # number of time steps
    self.h = self.tf / self.N       # time-step

  def f(self, k, x, u):
    # arm discrete dynamics
    # set jacobians A, B to [] if unavailable

    q = x[:2]
    v = x[2:4]

    c1 = np.cos(q[0])
    c2 = np.cos(q[1])
    s2 = np.sin(q[1])
    c12 = np.cos(q[0]+q[1])

    # coriolis matrix
    C = -self.m2*self.l1*self.lc2*s2*np.array([[v[1], v[0]+v[1]], [-v[0], 0]]) \
      + np.diag([0.2, 0.2])

    # mass elements
    m11 = self.m1*self.lc1**2 + self.m2*(self.l1**2 + self.lc2**2 + 
      2*self.l1*self.lc2*c2) + self.I1 + self.I2
    m12 = self.m2*(self.lc2**2 + self.l1*self.lc2*c2) + self.I2
    m22 = self.m2*self.lc2**2 + self.I2

    # mass matrix
    M = np.array([[m11, m12], [m12, m22]])

    # gravity vector
    fg = np.array([(self.m1*self.lc1 + self.m2*self.l1)*self.g*c1 + 
      self.m2*self.lc2*self.g*c12, self.m2*self.lc2*self.g*c12])

    # acceleration
    a = np.linalg.inv(M) @ (u - C@v - fg)
    v = v + self.h*a
    
    x = np.concatenate((q + self.h*v, v))

    # leave empty to use finite difference approximation
    A = np.array([])
    B = np.array([])

    return x, A, B


if __name__ == '__main__':

  arm = Arm()

  # initial state
  x0 = np.zeros(4)

  # controls
  us = np.zeros((2, arm.N))

  # states
  xs = np.zeros((4, arm.N+1))
  xs[:, 0] = x0

  for k in range(arm.N):
    xs[:, k+1], _, _ = arm.f(k, xs[:, k], us[:, k])

  plt.figure()
  plt.plot(xs[0,:], xs[1,:], '-b')
  plt.show()