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

    def __init__(self, num_measurments: 2):

        # hx() measurement function - convert state into a measurement
        # where measurements are [x_pos, y_pos]
        def hx(x):
            return np.array([x[0], x[2]])

        # fx() state transition function - predict next state based
        # on constant velocity model x = vt + x_0
        def fx(x, dt):
            F = np.array(
                [[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]], dtype=float
            )
            return np.dot(F, x)

        ## Generates sigma points and weights
        points = MerweScaledSigmaPoints(4, alpha=0.1, beta=2.0, kappa=-1)

        self.filter = UnscentedKalmanFilter(
            dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points
        )

        self.filter.P *= 0.5  # initial uncertainty

        z_std = 0.1

        self.filter.R = np.diag([z_std ** 2, z_std ** 2])  # measurment noise matrix

        self.filter.Q = filterpy.common.Q_discrete_white_noise(
            dim=2, dt=dt, var=0.01 ** 2, block_size=2
        )  # process noise matrix

    def set_initial_state(self, state_array):
        ##state_array : a numpy array of dim_x representing initial state
        self.filter.x = state_array
        self.filter.predict()

    def update_state(self, state_array):
        ##state_array : a numpy array of dim_z representing initial state
        self.filter.update(state_array)

    def track_object_offline(self, measurments):
        """
        measurments: a numpy array of shape (n_measurments, 1 + dim_z)

        This runs offline tracking on a set of measurments
        the required input is an array called mesaurments
        of shape (n_measurments, 1 + dim_z) i+dimz is a booleen
        representation a detection and dim_z is the detection
        """
        return
