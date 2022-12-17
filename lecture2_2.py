import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize

# Armijo step rule
def armijo(x, fx, gx, d):

  sigma = 0.01
  beta = 0.25
  s = 1

  f = fx(x)
  G = gx(x)
  for m in range(10):
    a = beta**m*s
    if (f - fx(x + a*d) > -sigma*beta**m*s*np.dot(G, d)):
      return a
  return a

# gradient step
def gstep(x, fx, gx):

  # gradient
  g = gx(x)

  # test for convergence
  if np.linalg.norm(g) < 1e-8:
    return x

  # direction
  d = -g

  # step-size
  a = armijo(x, fx, gx, d)

  # update state
  x = x + a*d

  return x
  
# gradient step
def Nstep(x, fx, gx, Hx, reg):

  # gradient
  g = gx(x)

  # hessian
  H = Hx(x)

  # test for convergence
  if np.linalg.norm(g) < 1e-8:
    return x

  # regularize (if enabled)
  if reg:
    #check for positive definiteness  
    e = np.linalg.eigvals(H)
    if (np.amin(e) < 0):
      # apply a trust-region like Hessian modification
      H = H + np.eye(2)*(np.abs(np.amin(e)) + .001)

  # direction
  d = np.linalg.solve(-H, g)

  # step-size
  a = armijo(x, fx, gx, d)

  # update state
  x = x + a*d

  return x


if __name__ == '__main__':
  # nonlinear function of two variables (symbolic)
  x1,x2 = sp.symbols('x1,x2')
  x = sp.Matrix([[x1],[x2]])
  f1 = x1*sp.exp(-(x1**2 + x2**2)) + (x1**2 + x2**2)/20
  f = sp.Matrix([f1])

  # compute gradient and hessian symbolically
  gradf = f.jacobian(x)
  hessf = gradf.jacobian(x)

  # handles for the function, it's gradient, and hessian
  fx = sp.lambdify([x], np.squeeze(f), 'numpy')
  _gx = sp.lambdify([x], gradf)
  gx = lambda x : np.squeeze(_gx(x))
  _Hx = sp.lambdify([x], hessf)
  Hx = lambda x : np.squeeze(_Hx(x))

  # Plot function
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  x1 = np.arange(-2, 2, 0.01)
  x2 = np.arange(-2, 2, 0.01)
  X1, X2 = np.meshgrid(x1, x2)
  Z = np.zeros((X1.shape[0], X1.shape[1]))
  for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
      Z[i, j] = fx(np.array([X1[i, j], X2[i, j]]))
  surface = ax.plot_surface(X1, X2, Z, cmap=cm.jet, linewidth=0, 
    antialiased=False)
  contour = ax.contour(X1, X2, Z, zdir='z', offset=-0.4, cmap=cm.jet)
  plt.title("$x_1*e^{-(x_1^2 + x_2^2)} + (x_1^2 + x_2^2)(1/20)$")
  plt.xlabel("$x_1$")
  plt.ylabel("$x_2$")

  reg = 1;  # whether to use regularized-Newton or not
  x0 = np.array([1.6, 1.9]); # reg-Newton and gradient converge; Newton is stuck
  # x0=np.array([-0.4, 0.6]); # closer to optimum: reg-Newton and gradient converge
  # x0=np.array([1, 0.1]);  # gradient get stuck

  # Test gradient method with Armijo rule
  x=x0
  xs = [x]
  for k in range(100):
    x = gstep(x, fx, gx); 
    xs.append(x)
  xs = np.array(xs)

  # plot result
  fig = plt.figure()
  ax = fig.add_subplot(111)
  contour = ax.contour(X1, X2, Z, cmap=cm.jet)
  plt.plot(xs[:,0], xs[:,1], '-bo', fillstyle='none')
  print("Gradient method: x = ", x, " f = ", fx(x))

  # Test newton method with Armijo rule
  x=x0
  xs = [x]
  for k in range(20):
    x = Nstep(x, fx, gx, Hx, reg); 
    xs.append(x)
  xs = np.array(xs)

  # plot result
  plt.plot(xs[:,0], xs[:,1], '-mo', fillstyle='none')
  plt.title("$x_1*e^{-(x_1^2 + x_2^2)} + (x_1^2 + x_2^2)(1/20)$")
  plt.xlabel("$x_1$")
  plt.ylabel("$x_2$")
  print("Newton method: x = ", x, " f = ", fx(x))

  # run a standard algorithm
  res = minimize(fx, x0, jac=gx)
  print("Scipy minimize: x = ", res.x, " f = ", fx(res.x))

  plt.show()