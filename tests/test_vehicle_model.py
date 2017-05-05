import numpy as np
from gym_offroad_nav.offroad_map import OffRoadMap
from gym_offroad_nav.vehicle_model.cython_impl import c_step
from tests.utils import datadir, load_test_case, run_function_with_test_case

class TestVehicleModel():

    def test_vehicle_model_cython(self, datadir):
        for i in range(8):
            test_case = load_test_case(datadir, "test_case_%d.pkl" % i)
            run_function_with_test_case(c_step, test_case)
