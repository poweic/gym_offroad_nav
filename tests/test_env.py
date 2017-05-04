import gym
import numpy as np
import gym_offroad_nav.envs

class TestOffRoadNavEnv():

    env = gym.make("OffRoadNav-v0")
    seed = 29979

    def test_observation_space(self):
        state = self.env.reset()

        # front view should be batch_size x height x width x channels
        assert state['front_view'].ndim == 4

        # vehicle_state should be batch_size x 6
        assert state['vehicle_state'].ndim == 2
        assert state['vehicle_state'].shape[1] == 6
