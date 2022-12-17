# Imports
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import cm


class HW1_Q3(object):
    """ Homework 1 Question 3 """

    def eval_L1(self, x):
        """Evaluate L(x)=(1-x1)^2+200*(x2-x1^2)**2"""
        """Input: x (numpy.ndarray, size: 2)"""
        """Output: L (numpy.float64)"""
        # 3a) TODO: Implement me
        x1 = x[0]
        x2 = x[1]
        L = (1 - x1)**2 + 200 * (x2 - x1**2) ** 2

        return L

    def eval_G1(self, x):
        """Evaluate gradient of L(x)=(1-x1)^2+200*(x2-x1^2)**2"""
        """Input: x (numpy.ndarray, size: 2)"""
        """Output: G (numpy.ndarray, size: 2)"""
        # 3a) TODO: Implement me
        # nonlinear function of two variables (symbolic)
        x1, x2 = sp.symbols('x1,x2')
        X = sp.Matrix([[x1], [x2]])
        f1 = (1 - x1)**2 + 200 * (x2 - x1**2) ** 2
        f = sp.Matrix([f1])

        # compute gradient and hessian symbolically
        gradf = f.jacobian(X)
        _gx = sp.lambdify([X], gradf)
        gx = lambda x: np.squeeze(_gx(x))
        G = gx(x)
        return G

    def eval_H1(self, x):
        """Evaluate hessian of L(x)=(1-x1)^2+200*(x2-x1^2)"""
        """Input: x (numpy.ndarray, size: 2)"""
        """Output: H (numpy.ndarray, size: 2x2)"""
        # 3a) TODO: Implement me
        # nonlinear function of two variables (symbolic)
        x1, x2 = sp.symbols('x1,x2')
        X = sp.Matrix([[x1], [x2]])
        f1 = (1 - x1) ** 2 + 200 * (x2 - x1 ** 2) ** 2
        f = sp.Matrix([f1])

        # compute hessian symbolically
        gradf = f.jacobian(X)
        hessf = gradf.jacobian(X)
        _Hx = sp.lambdify([X], hessf)
        Hx = lambda x: np.squeeze(_Hx(x))

        H = Hx(x)
        return H

    def armijo(self, x, fx, gx, d):
        """Calculate step size using armijo's rule"""
        """Inputs: x (numpy.ndarray, size: 2)
               fx (loss function)
               gx (gradient function)
               d (descent direction, numpy.ndarray, size: 2) """
        """Output: step (numpy.float64)"""
        # 3b) TODO: Implement me

        # define initial stepsize
        sigma = 0.01
        beta = 0.25
        s = 1

        # compute loss function and gradient
        f = fx(x)
        G = gx(x)
        for m in range(20):
            a = (beta ** m) * s
            if (f - fx(x + a * d) > -sigma * beta ** m * s * np.dot(G, d)) or a < 1e-8:
                return a

    def gradient_descent(self, x0, fx, gx):
        """Perform gradient descent"""
        """Inputs: x0 (numpy.ndarray, size: 2), 
               fx (loss function), 
               gx (gradient function)"""
        """Outputs: xs (numpy.ndarray, size: (numSteps+1)x2)"""
        # 3c) TODO: Implement me
        xs = [x0]
        x = x0
        for k in range(500000):
            # gradient
            g = gx(x)

            # test for convergence
            # if np.linalg.norm(g) < 1e-8:
            #     return np.array(xs)

            # direction
            d = -g

            # step-size
            a = self.armijo(x, fx, gx, d)
            # print(a)

            # test for step size
            if a < 1e-8:
                return np.array(xs)

            # update state
            x = x + a * d

            xs.append(x)
            k += 1
        return np.array(xs)

    def newton_descent(self, x0, fx, gx, hx):
        """Perform gradient descent"""
        """Inputs: x0 (numpy.ndarray, size: 2), 
               fx (loss function), 
               gx (gradient function)
               hx (hessian function)"""
        """Outputs: xs (numpy.ndarray, size: (numSteps+1)x2)"""
        # 3d) TODO: Implement me
        xs = [x0]
        x = x0
        for k in range(500000):
            # gradient
            g = gx(x)
            H = hx(x)

            # test for convergence
            # if np.linalg.norm(g) < 1e-8:
            #     return np.array(xs)

            # check for positive definiteness
            # e = np.linalg.eigvals(H)
            # if np.amin(e) < 0:
            #     # trust region
            #     H = H + np.eye(2)*(np.abs(np.amin(e)) + .001)

            # direction
            d = np.linalg.solve(-H, g)

            # step-size
            a = self.armijo(x, fx, gx, d)
            # print(a)

            # test for step size
            if a < 1e-8:
                return np.array(xs)

            # update state
            x = x + a * d

            xs.append(x)
            k += 1
        return np.array(xs)


    def eval_L2(self, x):
        """Evaluate L(x) = x1*exp(-x1^2-(1/2)*x2^2) + x1^2/10 + x2^2/10"""
        """Input: x (numpy.ndarray, size: 2)"""
        """Output: L (numpy.float64)"""
        x1 = x[0]
        x2 = x[1]
        L = x1*np.exp(-x1**2-(1/2)*x2**2) + x1**2/10 + x2**2/10
        return L

    def eval_G2(self, x):
        """Evaluate gradient of L(x) = x1*exp(-x1^2-(1/2)*x2^2) + x1^2/10 + x2^2/10"""
        """Input: x (numpy.ndarray, size: 2)"""
        """Output: G (numpy.ndarray, size: 2)"""
        x1, x2 = sp.symbols('x1,x2')
        X = sp.Matrix([[x1], [x2]])
        f1 = x1*sp.exp(-x1**2-(1/2)*x2**2) + x1**2/10 + x2**2/10
        f = sp.Matrix([f1])

        # compute gradient and hessian symbolically
        gradf = f.jacobian(X)
        _gx = sp.lambdify([X], gradf)
        gx = lambda x: np.squeeze(_gx(x))
        G = gx(x)
        return G

    def eval_H2(self, x):
        """Evaluate hessian of L(x) = x1*exp(-x1^2-(1/2)*x2^2) + x1^2/10 + x2^2/10"""
        """Input: x (numpy.ndarray, size: 2)"""
        """Output: H (numpy.ndarray, size: 2x2)"""
        x1, x2 = sp.symbols('x1,x2')
        X = sp.Matrix([[x1], [x2]])
        f1 = x1*sp.exp(-x1**2-(1/2)*x2**2) + x1**2/10 + x2**2/10
        f = sp.Matrix([f1])

        # compute hessian symbolically
        gradf = f.jacobian(X)
        hessf = gradf.jacobian(X)
        _Hx = sp.lambdify([X], hessf)
        Hx = lambda x: np.squeeze(_Hx(x))

        H = Hx(x)
        return H


if __name__ == '__main__':
    """This code runs if you execute this script"""
    hw1_q3 = HW1_Q3()

    # # TODO: Uncomment this line to visualize gradient descent & newton descent 
    # #       for first loss function
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x1 = np.arange(-2, 2, 0.01)
    x2 = np.arange(-2, 2, 0.01)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros((X1.shape[0], X1.shape[1]))
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = hw1_q3.eval_L1(np.array([X1[i, j], X2[i, j]]))
    contour = ax.contour(X1, X2, Z, cmap=cm.viridis)
    xsg = hw1_q3.gradient_descent(np.array([0, 0]), hw1_q3.eval_L1, hw1_q3.eval_G1)
    xsn = hw1_q3.newton_descent(np.array([0, 0]), hw1_q3.eval_L1, hw1_q3.eval_G1, hw1_q3.eval_H1)
    plt.plot(xsg[:, 0], xsg[:, 1], '-o')
    plt.plot(xsn[:, 0], xsn[:, 1], '-o')
    print("First loss function gradient method steps: ", xsg.shape[0]-1)
    print("First loss function newton method steps:   ", xsn.shape[0]-1)

    # # TODO: Uncomment this line to visualize gradient descent & newton descent
    # #       for second loss function
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x1 = np.arange(-3.5, 3.5, 0.01)
    x2 = np.arange(-3.5, 3.5, 0.01)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros((X1.shape[0], X1.shape[1]))
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = hw1_q3.eval_L2(np.array([X1[i, j], X2[i, j]]))
    contour = ax.contour(X1, X2, Z, cmap=cm.viridis)
    xsg = hw1_q3.gradient_descent(np.array([1.5, 3.0]), hw1_q3.eval_L2,
        hw1_q3.eval_G2)
    print("-"*50)
    xsn = hw1_q3.newton_descent(np.array([1.5, 3.0]), hw1_q3.eval_L2,
        hw1_q3.eval_G2, hw1_q3.eval_H2) # Not quite working yet
    plt.plot(xsg[:, 0], xsg[:, 1], '-o')
    plt.plot(xsn[:, 0], xsn[:, 1], '-o')
    print("Second loss function gradient method steps: ", xsg.shape[0]-1)
    print("Second loss function newton method steps:   ", xsn.shape[0]-1)
    plt.show()
