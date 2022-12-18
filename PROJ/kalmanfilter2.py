import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
class KF:
    def __init__(self, timesteps):
        self.dt = timesteps
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([0, 0., 0, 0.])
        self.kf.F = np.array([[1., self.dt, 0, 0],
                            [0., 1., 0, 0],
                            [0, 0, 1., self.dt],
                             [0., 0, 0, 1.]])
        self.kf.H = np.array([[1, 0., 0, 0],
                            [0, 0, 1, 0]])
        self.kf.P = np.eye(4)*0.01
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=0.1, block_size=2)
        self.kf.R = np.array([[0.005, 0],
                            [0, 0.005]])


    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.update(measured)
        self.kf.predict()
        x = self.kf.x[0]
        y = self.kf.x[2]
        result = np.array([x, y])
        return result

if __name__ == '__main__':

    def os_mearsurement(start, end, num, sigma=0.1, type_='line'):
        if type_ == 'line':
            os_p_move = np.linspace(start, end, num)
        for i in range(num):
            os_p_move[i, :] += np.random.randn(2) * sigma
        return os_p_move
    N=20
    os_move = os_mearsurement([-2, -5],[-1, 6],N,sigma=0.001)
    os_predict = np.zeros_like(os_move)
    f = KF(0.2)
    f.kf.x=np.asarray([])
    Ps = np.zeros((4, 4, N + 1))  # estimated covariances
    ts = np.zeros((N + 1, 1))  # times
    ts[0] = 0
    Ps[:, :, 0] = f.kf.P
    print('\n')
    for i in range(np.size(os_move,axis=0)):
        ts[i + 1] = i * f.dt
        z = os_move[i,:]
        # print('measured states', z, '\n')
        pre = f.predict(z[0],z[1])
        os_predict[i,:] = pre
        # print('predicted states: ', pre)
        Ps[:, :, i + 1] = f.kf.P_post
        print('post_p:\n ', f.kf.P_post)

    fig, axs = plt.subplots(1,3)
    axs[0].scatter(os_move[:, 0], os_move[:, 1],c='b',s=120, label="measured")
    axs[0].scatter(os_predict[:, 0], os_predict[:, 1], c='r',label="predicted")
    handles, labels = axs[0].get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    axs[0].legend(handles, labels)

    axs[1].plot(ts, np.reshape(np.sqrt(Ps[0, 0, :]), Ps.shape[2]),label="sigma_x")
    handles, labels = axs[1].get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    axs[1].legend(handles, labels)

    axs[2].plot(ts, np.reshape(np.sqrt(Ps[2, 2, :]), Ps.shape[2]), label="sigma_y")
    handles, labels = axs[2].get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    axs[2].legend(handles, labels)

    plt.show()
    print('end')
