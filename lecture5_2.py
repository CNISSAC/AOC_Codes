import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def traj(t, q0, v0, c2, c3):

  q = c3*t**3 + c2*t**2 + v0*t + q0
  v = 3*c3*t**2 + 2*c2*t + v0
  return (q, v)


def coeff(c, q0, v0, vf, tf):
  # implicit equation for the trajectory coefficient and constraint multiplier

  c2 = c[0:2]
  c3 = c[2:4]
  nu1 = c[4]

  (q, v) = traj(tf, q0, v0, c2, c3)

  f = np.concatenate((2*q*nu1 - 6*c3, [np.transpose(q)@q-1], v - vf))
  return f


def compute(q0, v0, vf, tf, N):

  # quess for initial conditions
  c2 = np.array([1, 1])
  c3 = np.array([1, 1])
  nu1 = 1
  c = np.array([c2[0], c2[1], c3[0], c3[1], nu1])

  coeff_ = lambda c_: coeff(c_, q0, v0, vf, tf)
  c = fsolve(coeff_, c)

  c2 = c[0:2]
  c3 = c[2:4]

  h = tf/N
  xs = np.zeros((4, N+1))

  for i in range(N+1):
    t = i*h
    (q, v) = traj(t, q0, v0, c2, c3)
    xs[:, i] = np.concatenate((q, v))

  return xs


if __name__ == '__main__':

  # draw circle
  circle = plt.Circle((0, 0), 1, color='b', fill=False)
  fig, ax = plt.subplots()
  ax.add_patch(circle)

  N = 50 # time-steps

  # generate 10 initial conditions and compute the optimal trajectories
  for i in range(10):
    q0 = np.random.normal(size=2)*3
    v0 = np.random.normal(size=2)*5
    vf = np.zeros(2)
    tf = 1

    xs = compute(q0, v0, vf, tf, N)
    plt.plot(xs[0, :], xs[1, :], '.g')

  plt.show()