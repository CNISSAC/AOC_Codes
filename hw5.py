import numpy as np
import matplotlib.pyplot as plt

class HW5_Q3(object):

    def __init__(self):
        self.size = 1000000
        self.theta_1 = np.random.rand(self.size) *2*np.pi-np.pi
        self.theta_2 = np.random.rand(self.size) *2*np.pi-np.pi
        self.l1 = 1
        self.l2 = 1
        self.p0 = np.array([0.5, 0.5])
        self.radius = 0.25
        self.S_1 = np.array((0, 0))
        self.A_1 = self.p0


    def Pt(self,theta_1, theta_2):

        pt = np.array((np.cos(theta_1)*self.l1 + np.cos(theta_1+theta_2)*self.l2,
                      np.sin(theta_1)*self.l1 + np.sin(theta_1+theta_2)*self.l2))
        return pt

    def P0(self, px, py):
        p0 = np.array([px, py])
        return p0

    def D1(self, theta_1):
        D_1 = np.array((np.cos(theta_1)*self.l1, np.sin(theta_1)*self.l1))
        return D_1

    def S2(self, theta_1):
        S_2 = self.D1(theta_1)
        return S_2

    def D2(self,theta_1, theta_2):
        D_2 = np.array((np.cos(theta_1+theta_2)*self.l2,
                     np.sin(theta_1+theta_2)*self.l2))
        return D_2

    def A2(self,theta_1):
        A_2 = self.p0 - self.D1(theta_1)
        return A_2

    def f(self, S, A, D, l):

        f_ = S + max(min(np.transpose(A)@(D/np.linalg.norm(D)), l), 0)*(D/np.linalg.norm(D))
        return f_


if __name__ == '__main__':
    hw5_q3 =HW5_Q3()
    theta_1 = hw5_q3.theta_1
    theta_2 = hw5_q3.theta_2
    p0 = hw5_q3.p0
    rdius = hw5_q3.radius

    # question (f)
    confi_1 = np.zeros((2, hw5_q3.size))
    for i in range(hw5_q3.size):
        pt = hw5_q3.Pt(theta_1[i], theta_2[i])
        if np.transpose(pt-p0)@ (pt-p0) >= rdius**2:
            confi_1[0, i] = theta_1[i]
            confi_1[1, i] = theta_2[i]

    plt.figure()
    plt.scatter(confi_1[0, :], confi_1[1, :], s=1)
    plt.xlabel("θ1")
    plt.ylabel("θ2")

    # question (g)
    confi_2 = np.zeros((2, hw5_q3.size))
    S_1 = hw5_q3.S_1
    A_1 = hw5_q3.A_1
    l_1 = hw5_q3.l1
    l_2 = hw5_q3.l2
    for i in range(hw5_q3.size):
        D_1 = hw5_q3.D1(theta_1[i])
        S_2 = hw5_q3.S2(theta_1[i])
        D_2 = hw5_q3.D2(theta_1[i], theta_2[i])
        A_2 = hw5_q3.A2(theta_1[i])
        f_1 = hw5_q3.f(S_1, A_1, D_1, l_1)
        f_2 = hw5_q3.f(S_2, A_2, D_2, l_2)
        if np.transpose(f_1-p0) @ (f_1-p0) >= rdius**2 and np.transpose(f_2-p0) @ (f_2-p0) >= rdius**2:
            confi_2[0, i] = theta_1[i]
            confi_2[1, i] = theta_2[i]
    plt.figure( )
    plt.scatter(confi_2[0, :], confi_2[1, :], s=1)
    plt.xlabel("θ1")
    plt.ylabel("θ2")




    plt.show()