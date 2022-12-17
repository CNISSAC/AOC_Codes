import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


if __name__ == '__main__':

  t,x0,tf = sp.symbols('t,x0,tf', real=True)

  # Solve P(t)
  P = sp.Function("P")
  diffeq_P = sp.Eq(sp.Derivative(P(t), t), P(t)**2 - 2 * P(t) - 1)
  Pt = sp.dsolve(diffeq_P, P(t), hint='1st_rational_riccati', ics={P(tf): 0}, 
    simplify=True)
  Pt = Pt.rhs

  # solve x(t) using:
  # l = P*x
  # u = -2*l
  # dx = x + u  and hence dx = (1-2*P)x
  x = sp.Function("x")
  diffeq_x = sp.Eq(sp.Derivative(x(t), t), (1 - 2*Pt)*x(t))
  xt = sp.dsolve(diffeq_x, x(t), ics={x(0): x0}, simplify=True)
  xt = xt.rhs

  x0_ = -1
  tf_ = 5
  t_ = np.arange(0, tf_, 0.01)
  fig, axs = plt.subplots(1, 3)

  Pt_func = sp.utilities.lambdify([t, x0, tf], Pt, 'numpy')

  Ps = Pt_func(t_, x0_, tf_)
  axs[0].plot(t_, Ps)
  axs[0].set_xlabel('t')
  axs[0].set_ylabel('P(t)')

  xt_func = sp.utilities.lambdify([t, x0, tf], xt, 'numpy')

  xs = xt_func(t_, x0_, tf_)
  axs[1].plot(t_, -2*Ps*xs)
  axs[1].set_xlabel('t')
  axs[1].set_ylabel('u(t)')

  xs = xt_func(t_, x0_, tf_)
  axs[2].plot(t_, xs)
  axs[2].set_xlabel('t')
  axs[2].set_ylabel('x(t)')

  plt.subplots_adjust(wspace=0.6)
  plt.show()