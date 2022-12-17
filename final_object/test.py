import numpy as np
import matplotlib.pyplot as plt
from DDP_moving_obstacles import DDP
from kalmanfilter import KalmanFilter

DDP = DDP()
TF = 2
n = DDP.N_seg
N_all = TF//DDP.tf*n

os_p_move = np.stack((np.linspace([-1, 6], [-2, -5], N_all+n), np.linspace([2, -5], [1, 10], N_all+n)), axis=0)
os_r_move = [0.25, 2]
os_p_predicted = np.zeros_like(os_p_move)  # predicted positions of moving obstacles
os_p = [[-2.5, 2], [-1, 0]]
os_r = [0.25, 0]
x0 = np.array([-10, -5, 1.2, 0, 0])  # initial state
xs = np.zeros((np.size(x0), N_all+1))
xs[:, 0] = x0

for k in range(0, TF, DDP.t_seg):
    for i in range(np.size(os_p_move, axis=0)):
        kf = KalmanFilter()
        os_p_last = []
        predicted = []
        for j in range(DDP.N_seg):
            predicted = kf.predict(os_p_move[i, k*n+j, 0], os_p_move[i, k*n+j, 1])  # using previous measurement to train KF
        for j in range(DDP.N_seg):
            predicted = kf.predict(predicted[0], predicted[1])
            os_p_predicted[i, n+k*n+j, :] = np.concatenate((predicted[0], predicted[1]), axis=0)
            os_p_last = os_p_predicted[i, n+k*n+j, :]  # store the last position of the obstacle
        os_p_predicted[i, n+k*n+DDP.N_seg:, :] = np.full_like(os_p_move[i, n+k*n+DDP.N_seg:, :], os_p_last)

    os_p_m = os_p_predicted[:, n + k * n:, :]
    # os_p_m = os_p_move[:, n+k*n:, :]
    t = TF-k*DDP.t_seg
    xs_, N = DDP.trajectory(t, x0, os_p, os_r, os_p_m, os_r_move )
    xs[:, k*n+1: (k+1)*n+1] = xs_[:, 1:n+1]
    x0=xs_[:, n]


fig, axs = plt.subplots()
plt.axis([-10, 10, -10, 15])

for j in range(len(os_r)):
    circle = plt.Circle(os_p[j], os_r[j], color='r', linewidth=2,
                    fill=False)
    axs.add_patch(circle)
size = np.size(os_p_move,1)
for k in range(np.size(os_p_move,1)):
    axs.axis('equal')
    plt.axis([-15, 15, -10, 15])

    if(k >= n):
        plt.scatter(xs[0, k-n], xs[1, k-n])
    for j in range(len(os_r_move)):
        circle_true = plt.Circle(os_p_move[j][k], os_r_move[j], color='r', linewidth=2,
                      fill=False)
        circle_predicted = plt.Circle(os_p_predicted[j][k], os_r_move[j], color='b', linewidth=2,
                      fill=False)
        axs.add_patch(circle_true)
        axs.add_patch(circle_predicted)
    plt.pause(DDP.h)
    if k != np.size(os_p_move,1)-1:
        for j in range(len(os_r_move)):
            axs.patches.pop()
            axs.patches.pop()


plt.show()
