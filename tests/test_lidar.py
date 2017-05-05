import numpy as np
from gym_offroad_nav.lidar.lidar import c_lidar_mask
from tests.utils import datadir, load_test_case, run_function_with_test_case

class TestLidar():

    def test_lidar_cython(self, datadir):
        for i in range(2):
            test_case = load_test_case(datadir, "test_case_%d.pkl" % i)
            run_function_with_test_case(c_lidar_mask, test_case)
