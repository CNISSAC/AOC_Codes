import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

import warnings
warnings.simplefilter("ignore", RuntimeWarning)


def angle(a, p):
  s0 = 1.0 / np.cos(a[0])
  sf = 1.0 / np.cos(a[1])
  t0 = np.tan(a[0])
  tf = np.tan(a[1])
  f = p - np.array([(sf*(tf-t0)-t0*(sf-s0)+np.arcsinh(tf)-np.arcsinh(t0))/2.0,
                    s0 - sf])
  return f


def zermelo(t, q, V):
  x = q[0]
  y = q[1]
  a = q[2]

  u = -V*y

  f = np.array([np.cos(a)*V+u, np.sin(a)*V, np.cos(a)**2*V])
  return f


if __name__ == '__main__':

  p = np.array([3.66, -1.86])

  a = np.array([105.0*np.pi/180.0, 240.0*np.pi/180.0])

  angle_ = lambda a_: angle(a_, p)
  a = fsolve(angle_, a)

  a = np.array([105.0*np.pi/180.0, 240.0*np.pi/180.0])

  V = 0.3

  q = np.array([p[0], p[1], a[0]])

  zermelo_ = lambda t_, q_: zermelo(t_, q_, V)
  result = solve_ivp(zermelo_, (0, 18), q, method='RK45', 
    t_eval = np.arange(0, 18, 0.1))
  y = result.y

  plt.figure()
  plt.quiver(y[0,0:-1:10], y[1,0:-1:10], V*np.cos(y[2,0:-1:10]), 
    V*np.sin(y[2,0:-1:10]),color='b')
  plt.plot(y[0,:], y[1,:],'-g')
  plt.show()