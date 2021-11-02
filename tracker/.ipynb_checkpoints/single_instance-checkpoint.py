from filterpy import Q_discrete_white_noise
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
import numpy as np


class SingleInstanceTracker:
    """
    This class implements single object tracker. Since there is a known single object we do not need to associate object IDs,
    but we need to filter points. This is becasue we assume our detections are noisy estimates of our objects location
    and/or keypoints.
    """

    def __init__(self):
        print("tracker initilized")

        self.dt = 0.1

        # create sigma points to use in the filter. This is standard for Gaussian processes
        self.points = MerweScaledSigmaPoints(4, alpha=0.1, beta=2.0, kappa=-1)

        self.kf = filterpy.kalman.UnscentedKalmanFilter(
            dim_x=4, dim_z=2, dt=self.dt, fx=self.fx, hx=self.hx, points=self.points
        )

        self.kf.x = np.array([-1.0, 1.0, -1.0, 1])  # initial state
        self.kf.P *= 0.1  # initial uncertainty
        self.z_std = 0.1
        self.kf.R = np.diag([z_std ** 2, z_std ** 2])  # 1 standard
        self.kf.Q = filterpy.common.Q_discrete_white_noise(
            dim=2, dt=dt, var=0.01 ** 2, block_size=2
        )

    def track_object_offline(self, trajectory):
        trajectory_filtered = []
        for z in trajectory:
            self.kf.predict()
            self.kf.update(z)
            # print(hx(kf.x), 'log-likelihood', kf.log_likelihood)
            trajectory_filtered.append(self.hx(self.kf.x))
        return trajectory_filtered

    def fx(self, x, dt):
        # state transition function - predict next state based
        # on constant velocity model x = vt + x_0
        F = np.array(
            [[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]], dtype=float
        )

        return np.dot(F, x)

    def hx(self, x):
        return np.array([x[0], x[2]])
