# Imports
import numpy as np
import matplotlib.pyplot as plt

class HW6_Q4(object):
  """ Homework 6 Questions 4 """


  def __init__(self):

    # 4a) TODO: Implement me
    self.dt = 0.1
    self.A = np.array([[1, self.dt], [self.dt/5, 1-self.dt/2]])
    self.B = np.array([[0], [self.dt]])
    self.w = np.array([[0], [self.dt/10]])
    self.Pf = np.array([[1, 0], [0, 1]])
    self.Q = np.zeros((2, 2))
    self.R = np.array([[0.04]])
    self.N = 100


  def calc_Ps_bs(self):
    """Calculate Ps and bs"""
    """Outputs: Ps (numpy.ndarray, shape: (2,2,101))
                bs (numpy.ndarray, shape: (2,1,101))"""

    # 4b) TODO: Implement me
    Ps = np.zeros((2, 2, self.N+1))
    bs = np.zeros((2, 1, self.N+1))
    Ps[:, :, self.N] = self.Pf
    bs[:, :, self.N] = np.zeros((2, 1))
    A = self.A
    B = self.B
    Q = self.Q
    R = self.R
    w = self.w
    A_T = np.transpose(A)
    B_T = np.transpose(B)
    w_T = np.transpose(w)

    for i in range(self.N-1, -1, -1):
      P_i_1 = Ps[:, :, i+1]
      P_i_1_T = np.transpose(P_i_1)
      b_i_1 = bs[:, :, i+1]
      b_i_1_T = np.transpose(b_i_1)
      Ps[:, :, i] = Q + A_T@(P_i_1-P_i_1_T@B@np.linalg.inv((R+B_T@P_i_1@B))@B_T@P_i_1)@A
      bs_T = (w_T@P_i_1+b_i_1_T)@(A-B@np.linalg.inv((R+B_T@P_i_1@B))@B_T@P_i_1@A)
      bs[:, :, i] = np.transpose(bs_T)



    return Ps, bs


  def control(self, x, P, b):
    """Calculate u_i given x_i, P_{i+1}, b_{i+1}"""
    """Inputs: x (numpy.ndarray, shape: (2))
               P (numpy.ndarray, shape: (2,2))
               b (numpy.ndarray, shape: (2,1))"""
    """Outputs: u (numpy.float64 or float or numpy.ndarray, shape: (1,1) or (1,))"""

    # 4c) TODO: Implement me
    A = self.A
    B = self.B
    R = self.R
    B_T = np.transpose(B)
    w = self.w
    K = -np.linalg.inv((R+B_T@P@B))@B_T@P@A
    k = -np.linalg.inv((R+B_T@P@B))@B_T@(b+P@w)
    u = K@x+k


    return u


  def dynamics(self, x, u):
    """Calculate x_{i+1} given x_i, u_i"""
    """Inputs: x (numpy.ndarray, shape: (2,))
               u (numpy.float64 or float or numpy.ndarray, shape: (1,1) or (1,))"""
    """Outputs: x_ (numpy.ndarray, shape: (2,))"""

    # 4c) TODO: Implement me
    # A = self.A
    # B = self.B
    # w = self.w
    # u = np.array([u])
    # iterm_2 = B@u
    # x_test = A@x+B@u+w@np.array([1])
    x_ = np.array([0.0, 0.0])
    p_i = x[0]
    v_i = x[1]
    p_i_1 = p_i + self.dt*v_i
    v_i_1 = v_i + self.dt*(-0.5*v_i+0.2*p_i+u+0.1)
    x_[0] = p_i_1
    x_[1] = v_i_1
    # if(x_test.all!=x_.all):
    #   print(x_test, x_)
    #   # quit()

    return x_


  def calc_xs_us(self, x, Ps, bs):
    """Calculate xs and us"""
    """Inputs: x (numpy.ndarray, size: 2)
               Ps (numpy.ndarray, size: 2x2x101)
               bs (numpy.ndarray, size: 2x1x101)"""
    """Outputs: xs (numpy.ndarray, size: 2x101)
                us (numpy.ndarray, size: 100)"""

    # 4c) TODO: Implement me
    xs = np.zeros((2, self.N+1))
    us = np.zeros((self.N))
    xs[:, 0] = x
    for i in range(self.N):
      us[i] = self.control(xs[:, i], Ps[:, :, i+1], bs[:, :, i+1])
      xs[:, i+1] = self.dynamics(xs[:, i], us[i])

    return xs, us


if __name__ == '__main__':
  """This code runs if you execute this script"""
  hw6_q4 = HW6_Q4()

  # # TODO: Uncomment the following lines to generate plots to visualize the 
  # result of your functions
  Ps, bs = hw6_q4.calc_Ps_bs()
  x = np.array([10, 0])
  xs, us = hw6_q4.calc_xs_us(x, Ps, bs)
  plt.figure()
  plt.plot(xs[0, :], xs[1, :])
  plt.xlabel("Position")
  plt.ylabel("Velocity")
  plt.title("xI = [10, 0]")
  x = np.array([10, 5])
  xs, us = hw6_q4.calc_xs_us(x, Ps, bs)
  plt.figure()
  plt.plot(xs[0, :], xs[1, :])
  plt.xlabel("Position")
  plt.ylabel("Velocity")
  plt.title("xI = [10, 5]")
  x = np.array([10, -5])
  xs, us = hw6_q4.calc_xs_us(x, Ps, bs)
  plt.figure()
  plt.plot(xs[0, :], xs[1, :])
  plt.xlabel("Position")
  plt.ylabel("Velocity")
  plt.title("xI = [10, -5]")
  plt.show()