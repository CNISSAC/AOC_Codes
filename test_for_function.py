import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize


# 测试矩阵
# x1 = np.arange(-2, 2, 0.01)
# x2 = np.arange(-2, 2, 0.01)
# x1_sample = x1[0:2]
# X1, X2 = np.meshgrid(x1, x2)

# 测试jacobian
x = np.array([0, 0])
x1, x2 = sp.symbols('x1,x2')
X = sp.Matrix([[x1], [x2]])
f1 = (1 - x1) ** 2 + 200 * (x2 - x1 ** 2) ** 2
f = sp.Matrix([f1])

# compute gradient and hessian symbolically
gradf = f.jacobian(X)
_gx = sp.lambdify([X], gradf)
gx = lambda x: np.squeeze(_gx(x))
G = gx(x)


e = [6, -6, 2, 0]
print(np.amin(e))