import numpy as np
import matplotlib.pyplot as plt
from ddp_traj import ddp_traj
from ddp_cost import ddp_cost
from ddp import ddp

# Optimal control example of a simple car model with disk obstacles
# Control bounds  were added

class Problem:

  def __init__(self):

    # time-step and # of segments
    self.tf =10
    self.N = 30
    self.h = self.tf/self.N
    
    # system mass
    self.m = 2

    # cost function specification
    self.Q = np.diag([0, 0, 0, 0, 0])
    self.R = np.diag([1, 5])
    self.Pf = np.diag([5, 5, 1, 1, 1])

    self.mu = 1

    self.os_p = [[-2.5, -2.5], [-1, 0]]
    self.os_r = [1.2, 1]
    self.ko_x = 2.7
    self.ko_u = 1

    # initial state
    self.x0 = np.array([-5, -2, -1.2, 0, 0])

    # control bound
    self.u_bd = np.array([0.5, 0.3])  # absolute control boundary


  def f(self, k, x, u):

    h = self.h
    c = np.cos(x[2])
    s = np.sin(x[2])
    v = x[3]
    w = x[4]

    A = np.array([[1, 0, -h * s * v, h * c, 0],
                  [0, 1, h * c * v, h * s, 0],
                  [0, 0, 1, 0, h],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])

    B = np.array([[0, 0],
                  [0, 0],
                  [0, 0],
                  [h, 0],
                  [0, h]])

    x = np.array([x[0] + h * c * v,
                  x[1] + h * s * v,
                  x[2] + h * w,
                  x[3] + h * u[0],
                  x[4] + h * u[1]])

    return x, A, B

  def L(self, k, x, u):

    if k < self.N:
      L = self.h * 0.5 * (np.transpose(x)@self.Q@x + np.transpose(u)@self.R@u)
      Lx = self.h * self.Q @ x
      Lxx = self.h * self.Q
      Lu = self.h * self.R @ u
      Luu = self.h * self.R
    else:
      L = np.transpose(x) @ self.Pf @ x * 0.5
      Lx = self.Pf @ x
      Lxx = self.Pf
      Lu = np.zeros(self.m)
      Luu = np.zeros((self.m, self.m))

    if k < self.N and hasattr(self, 'u_bd'):
      for i in range(len(u)):
        c_1 = np.abs(u[i])-self.u_bd[i]
        c_u = max(c_1, 0)
        v_u = np.array([np.sign(u[i]), 0])

        L = L + self.ko_u/2*c_u**2
        Lu = Lu + self.ko_u*v_u*c_u
        Luu = Luu + self.ko_u*v_u*np.transpose(v_u)*c_u/abs(c_1)

    if hasattr(self, 'os_r'):

      for i in range(len(self.os_r)):
        g = x[:2] - self.os_p[i]
        c_2 = self.os_r[i] - np.linalg.norm(g)
        v_x = -g/np.linalg.norm(g)
        c_x = max(c_2, 0)

        L = L + self.ko_x/2.0*c_x**2
        Lx[:2] = Lx[:2] + self.ko_x*v_x*c_x
        Lxx[:2, :2] = Lxx[:2, :2] + self.ko_x*v_x@np.transpose(v_x)*c_x/np.abs(c_2)

    return L, Lx, Lxx, Lu, Luu

if __name__ == '__main__':

  prob = Problem()

  # initial control sequence
  us = np.concatenate((np.tile([[0.1], [0.05]], (1, prob.N//2)), 
    np.tile([[-0.1], [-0.05]], (1, prob.N//2))), axis=1)/2

  xs = ddp_traj(us, prob)
  V = ddp_cost(xs, us, prob)

  fig, axs = plt.subplots(1, 2)
  axs[0].plot(xs[0, :], xs[1, :], '-b')
  for j in range(len(prob.os_r)):
    circle = plt.Circle(prob.os_p[j], prob.os_r[j], color='r', linewidth=2, 
      fill=False)
    axs[0].add_patch(circle)
  axs[0].axis('equal')


  for i in range(40):
    dus, V, Vn, dV, a = ddp(us, prob)
    # update control
    us = us + dus
    prob.a = a
    xs = ddp_traj(us, prob)
    axs[0].plot(xs[0,:], xs[1,:], '-g')
    prob.ko_x = prob.ko_x*1.1
    prob.ko_u = prob.ko_u*1.1
  axs[0].plot(xs[0,:], xs[1,:], '-m')

  axs[1].plot(np.arange(0, prob.tf, prob.h), us[0, :], color='b')
  axs[1].plot(np.arange(0, prob.tf, prob.h), us[1, :], color='g')
  axs[1].axhline(prob.u_bd[0], color='b', linestyle='--')
  axs[1].axhline(prob.u_bd[1], color='g', linestyle='--')
  axs[1].axhline(-prob.u_bd[0], color='b', linestyle='--')
  axs[1].axhline(-prob.u_bd[1], color='g', linestyle='--')
  axs[1].set_xlabel("sec.")
  axs[1].legend(["u_1", "u_2", 'u1_boundary', 'u2_boundary'], fontsize=8)
  

  plt.show()