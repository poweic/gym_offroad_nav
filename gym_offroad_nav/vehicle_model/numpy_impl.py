import os
import scipy.io
import numpy as np
from gym_offroad_nav.utils import dirname
from gym_offroad_nav.vehicle_model.cython_impl import c_step

class VehicleModel():

    def __init__(self, timestep, noise_level=0., wheelbase=2.0, drift=False):

        # Load vehicle model ABCD
        current_dir = dirname(__file__)
        model = scipy.io.loadmat(current_dir + "/vehicle_model_ABCD.mat")
        self.A = model["A"]
        self.B = model["B"]
        self.C = model["C"]
        self.D = model["D"]

        #
        self.timestep = timestep
        self.noise_level = noise_level
        self.wheelbase = wheelbase
        self.drift = drift
        self.rng = np.random.RandomState()

        if not drift:
            self.A[0][0] = 0
            self.C[0][0] = 0
            self.D[0][1] = 0

        # x = Ax + Bu, y = Cx + Du
        # Turn cm/s, degree/s to m/s and rad/s
        Q = np.diag([100., 180./np.pi])
        Qinv = np.diag([0.01, 0.01, np.pi/180.])
        self.B = self.B.dot(Q)
        self.C = Qinv.dot(self.C)
        self.D = Qinv.dot(self.D).dot(Q)

        # x is the unobservable hidden state, y is the observation
        # u is (v_forward, yaw_rate), y is (vx, vy, w), where
        # vx is v_slide, vy is v_forward, w is yaw rate
        # x' = Ax + Bu (prediction)
        # y' = Cx + Du (measurement)
        self.x = None

    def seed(self, rng):
        self.rng = rng

    def _predict(self, x, u):
        u = u.reshape(2, -1)
        y = np.dot(self.C, x) + np.dot(self.D, u)
        x = np.dot(self.A, x) + np.dot(self.B, u)
        return y, x

    def steer_to_yawrate(self, state, steer):
        vf = state[4]
        yawrate = vf * np.tan(steer) / self.wheelbase
        return yawrate

    def predict(self, state, action, n_sub_steps, map):

        action = action.astype(np.float64, order='C')
        random_seed = self.rng.randint(low=2, high=np.iinfo(np.uint32).max)

        # c_step implicitly ASSUME x, state, action, noise are contiguous array
        # with row-major memory layout (i.e. order='C')
        rewards, distances_traveled = c_step(
            self.x, state, action, n_sub_steps,
            self.timestep, self.noise_level, self.wheelbase, float(self.drift),
            random_seed,
            map.reward_map, dict(map.bounds), map.cell_size,
            low_speed_penalty=map.low_speed_penalty,
            decay_rate=map.low_speed_penalty_decay_rate,
            high_acc_penalty=map.high_acc_penalty
        )

        return state, rewards.reshape(-1), distances_traveled.reshape(-1)

    def predict_old(self, state, action):
        if self.x is None:
            raise ValueError("self.x is still None. Call reset() first.")

        # assumming all are column-vector
        assert state.shape[0] == 6, "state.shape = {}".format(state.shape)
        assert action.shape[0] == 2, "action.shape = {}".format(action.shape)
        assert self.x.shape[0] == 4, "self.x.shape = {}".format(self.x.shape)

        action = action.copy()
        action[1] = self.steer_to_yawrate(state, action[1])

        # y = state[3:6]
        y, self.x = self._predict(self.x, action)

        # theta is in radian
        theta = state[2]

        # Use uni-cycle model (this assume vehicle has no width)
        c, s = np.cos(theta), np.sin(theta)
        M = np.array([[c, -s], [s, c]])
        M = np.rollaxis(M, 2, 0)

        V = np.zeros_like(state[0:3])
        for i in range(V.shape[1]):
            V[0:2, i] = np.dot(M[i], state[3:5, i])
            V[2:3, i] += state[5:6, i]

        # dx = v * dt
        delta = V * self.timestep

        # Add some noise using delta * (1 + noise) instead of delta + noise
        white_noise = self.rng.randn(*delta.shape)
        delta *= 1 + white_noise * self.noise_level

        # x2 = x1 + dx
        state[0:3] += delta
        state[3:6] = y

        return state

    def reset(self, state, mask=None):
        # state: [x, y, theta, x', y', theta']
        # extract the last 3 elements from state
        y0 = state[3:6].reshape(3, -1)

        if mask is None:
            self.x = np.dot(np.linalg.pinv(self.C), y0)
        else:
            self.x[:, mask] = np.dot(np.linalg.pinv(self.C), y0[:, mask])
