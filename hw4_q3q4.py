# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from control import lqr

class HW4_Q3(object):
  """ Homework 4 Question 3 """

  def P2c(self, P):
    # convert from a 2x2 symmetric matrix to a 3x1 vector of its unique entries
    c = np.array([P[0,0], P[0,1], P[1,1]])
    return c

  def c2P(self, c):
    # the reverse of P2c
    P = np.array([[c[0], c[1]], [c[1], c[2]]])
    return P


  def __init__(self):
    """
    A: numpy.ndarray, shape: (2,2)
    B: numpy.ndarray, shape: (2,1)
    Q: numpy.ndarray, shape: (2,2)
    R: numpy.ndarray, shape: (1,1)
    tf: float
    Pf: numpy.ndarray, shape: (2,2)
    dt: float
    x0: numpy.ndarray, shape: (2,)
    """
    # 3a) TODO: Plug in A, B, Q, R, tf, Pf, dt, x0

    self.A = np.array([[0, 1], [2, -1]])
    self.B = np.array([[0], [1]])
    self.Q = np.array([[2, 0], [0,  1]])
    self.R = np.array([[1]])
    self.tf = 20
    self.Pf = np.zeros((2, 2))
    self.dt = 0.1
    self.x0 = np.array([-5, 5])


  def Riccati(self, t, c, A, B, Q, R):
    """Calculate Pdot using the Riccati ODE"""
    """Inputs: t (numpy.float64 or float)
               c (unique elements of P: [P[0,0], P[0,1], P[1,1]], 
                  numpy.ndarray, shape: (3,))
               A (numpy.ndarray, shape: (2,2))
               B (numpy.ndarray, shape: (2,1))
               Q (numpy.ndarray, shape: (2,2))
               R (numpy.ndarray, shape: (1,1))"""
    """Output: dc (unique elements of dP, numpy.ndarray, shape: (3,))"""
    # 3b) TODO: Implement me
    #   Note: use c2P and P2c functions to convert from vector c to matrix P
    P = self.c2P(c)
    dP = -P @ A - np.transpose(A) @ P - Q + P @ B @ np.linalg.inv(R) @ np.transpose(B) @ P
    dc = self.P2c(dP)
    return dc


  def integrate_Pt(self, A, B, Q, R, Pf, tf, dt):
    """Integrate P(t) from tf to 0 given dt using solve_ivp with RK45"""
    """Inputs: A (numpy.ndarray, shape: (2,2))
               B (numpy.ndarray, shape: (2,1))
               Q (numpy.ndarray, shape: (2,2))
               R (numpy.ndarray, shape: (1,1))
               Pf (numpy.ndarray, shape: (2,2))
               tf (float)
               dt (float)"""
    """Output: ts (numpy.ndarray, shape: (200,))
               Ps (numpy.ndarray, shape: (2,2,200))"""
    # 3c) TODO: Implement me
    cf = self.P2c(Pf)
    Riccati_ = lambda tf_, cf_: self.Riccati(tf_, cf_, A, B, Q, R)
    result = solve_ivp(Riccati_, (tf, 0), cf, method='RK45',
                       t_eval=np.arange(tf, 0, -dt))

    ts = np.flip(result.t)
    cs = np.flip(result.y, axis=1)  # cs shape: (3,200)
    Ps = np.zeros((2, 2, 200))
    for i in range(cs.shape[1]):
      Ps[:, :, i] = self.c2P(cs[:, i])  # Ps shape:(2,2,200)

    return ts, Ps


  def calc_u(self, x, P, B, R):
    """Calculate u given x, P, B, and R"""
    """Inputs: x (numpy.ndarray, shape: (2,))
               P (numpy.ndarray, shape: (2,2))
               B (numpy.ndarray, shape: (2,1))
               R (numpy.ndarray, shape: (1,1))"""
    """Output: u (numpy.ndarray, shape: (1,))"""
    # 3d) TODO: Implement me

    u = -np.linalg.inv(R) @ np.transpose(B) @ P @ x
    return u


  def calc_dx(self, x, u, A, B):
    """Calculate dx given x, u, A, and B"""
    """Inputs: x (numpy.ndarray, shape: (2,))
               u (numpy.ndarray, shape: (1,))
               A (numpy.ndarray, shape: (2,2))
               B (numpy.ndarray, shape: (2,1))"""
    """Output: dx (numpy.ndarray, shape: (2,))"""
    # 3e) TODO: Implement me

    dx = A @ x + B @ u
    return dx


  def solve_ut_xt(self, x0, A, B, R, dt, Ps):
    """Solve u(t) and x(t) given x0, A, B, R, and P(t)"""
    """Inputs: x (numpy.ndarray, size: 5)
               Ps (numpy.ndarray, size: 2x2x200)"""
    """Output: us (numpy.ndarray, size: 1x200)
               xs (numpy.ndarray, size: 2x201)"""
    # 3d) TODO: Implement me
    # Note: we recommend integrating manually as in
    #       lecture notes, instead of using a solver

    xs = np.zeros((2,201))
    us = np.zeros((1,200))
    xs[:, 0] = x0
    N = Ps.shape[2]
    for i in range(N):
      x = xs[:, i]
      P = Ps[:, :, i]
      u = self.calc_u(x, P, B, R)
      dx = self.calc_dx(x, u, A, B)
      xs[:, i+1] = dx*dt + x
      us[:, i] = u


    return us, xs

class HW4_Q4(object):
  """ Homework 4 Question 4 """


  def __init__(self):
    """
    Q: numpy.ndarray, shape: (2,2)
    R: numpy.ndarray, shape: (1,1)
    tf: float
    dt: float
    x0: numpy.ndarray, shape: (2,)
    """
    # 4a) TODO: Implement me
    self.Q = np.diag([5, 5, 0.01, 0.1, 0.1])
    self.R = np.diag([0.5, 0.1])
    self.tf =5
    self.dt = 0.01
    self.x0 = np.array([0, 0, 0, 0, 0])


  def calc_dx(self, x, u):
    """Calculate xdot"""
    """Inputs: x (numpy.ndarray, shape: (5,))
               u (numpy.ndarray, shape: (2,))"""
    """Output: dx (numpy.ndarray, shape: (5,))"""
    # 4b) TODO: Implement me
    dx = np.zeros(5)
    dx[0] = x[3]*np.cos(x[2])
    dx[1] = x[3]*np.sin(x[2])
    dx[2] = x[3]*np.tan(x[4])
    dx[3] = u[0]
    dx[4] = u[1]

    return dx


  def calc_de(self, x, u):
    """Calculate edot"""
    """Inputs: x (numpy.ndarray, shape: (5,))
               u (numpy.ndarray, shape: (2,))"""
    """Output: de (numpy.ndarray, shape: (5,))"""
    # 4c) TODO: Implement me

    dxd = np.array([1, 2, 0, 0, 0])
    dx = self.calc_dx(x, u)
    de = dx - dxd

    return de


  def calc_A_B(self, xd):
    """Calculate A and B"""
    """Inputs: xd (numpy.ndarray, shape: (5,))"""
    """Outputs: A (numpy.ndarray, shape: (5,5))
                B (numpy.ndarray, shape: (5,2))"""
    # 4d) TODO: Implement me

    A = np.zeros((5, 5))
    B = np.zeros((5, 2))
    A[0, 2] = -xd[3]*np.sin(xd[2])
    A[0, 3] = np.cos(xd[2])
    A[1, 2] = xd[3]*np.cos(xd[2])
    A[1, 3] = np.sin(xd[2])
    A[2, 3] = np.tan(xd[4])
    A[2, 4] = xd[3]/(np.cos(xd[4])**2)
    B[3, 0] = 1
    B[4, 1] = 1
    return A, B


  def solve_ut_xt(self, x0, Q, R, tf, dt):
    """Solve u(t) and x(t)"""
    """Output: us (numpy.ndarray, shape: (2,500))
               xs (numpy.ndarray, shape: (5,501))"""
    # 4e) TODO: Implement me
    # Note: we recommend integrating manually as in
    #       lecture notes, instead of using a solver

    us = np.zeros((2, 500))
    xs = np.zeros((5, 501))
    xs[:, 0] = x0
    e0 = np.array([0, 0, -np.arctan(2), -np.sqrt(5), 0])
    e_i = e0
    ud = np.array([0, 0])
    for i in range(int(tf/dt)):
      xd_i = np.array([i*dt, 2*i*dt, np.arctan(2), np.sqrt(5), 0])
      [A, B] = self.calc_A_B(xd_i)
      K = lqr(A, B, Q, R)[0]
      us[:, i] = -K @ e_i + ud
      de = self.calc_de(xs[:, i], us[:, i])
      e_i += de*dt
      # dx = self.calc_dx(xs[:, i], us[:, i])
      # xs[:, i+1] = dx*dt + xs[:, i]

      xs[:, i + 1] = np.array([(i+1) * dt, 2 * (i+1) * dt, np.arctan(2), np.sqrt(5), 0]) + e_i



    return us, xs

if __name__ == '__main__':
    """This code runs if you execute this script"""
    hw4_q3 = HW4_Q3()
    hw4_q4 = HW4_Q4()

    # # TODO: Uncomment the following lines to generate plots to visualize the result of your functions

    ts, Ps = hw4_q3.integrate_Pt(hw4_q3.A, hw4_q3.B, hw4_q3.Q, hw4_q3.R,
      hw4_q3.Pf, hw4_q3.tf, hw4_q3.dt)
    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for i in range(2):
      for j in range(2):
        axs[i, j].plot(ts, Ps[i, j, :])
        axs[i, j].set_xlabel("Time (s)")
        axs[i, j].set_ylabel("P"+str(i)+str(j)+"(t)")

    us, xs = hw4_q3.solve_ut_xt(hw4_q3.x0, hw4_q3.A, hw4_q3.B, hw4_q3.R,
      hw4_q3.dt, Ps)
    fig, axs = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    axs[0].plot(ts, us[0, :])
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("u(t)")
    axs[1].plot(ts, xs[0, :-1])
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("x0(t)")
    axs[2].plot(ts, xs[1, :-1])
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("x1(t)")

    us, xs = hw4_q4.solve_ut_xt(hw4_q4.x0, hw4_q4.Q, hw4_q4.R, hw4_q4.tf,
      hw4_q4.dt)
    plt.figure()
    plt.plot(xs[0, :], xs[1, :])
    xds = np.zeros((5, int(hw4_q4.tf/hw4_q4.dt)))
    for (i, t) in enumerate(np.arange(0, hw4_q4.tf, hw4_q4.dt)):
      xds[:, i] = np.array([t, 2*t, np.arctan2(2, 1), np.sqrt(5), 0])
    plt.plot(xds[0, :], xds[1, :], '--')
    plt.legend(["Car Path", "Desired Trajectory"])
    plt.xlabel("x0 (meters)")
    plt.ylabel("x1 (meters)")

    fig, axs = plt.subplots(1, 2)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    axs[0].plot(us[0, :])
    axs[0].set_title("Acceleration")
    axs[0].set_xlabel("Acceleration (m/s^2)")
    axs[0].set_ylabel("Time (s)")
    axs[1].plot(us[1, :])
    axs[1].set_title("Steering Rate")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Steering Rate (rad/s)")

    plt.show()