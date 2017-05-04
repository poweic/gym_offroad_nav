import os
import numpy as np
import scipy.io
import cPickle
from attrdict import AttrDict
from pytest import fixture
from distutils import dir_util
from gym_offroad_nav.offroad_map import OffRoadMap
# from gym_offroad_nav.vehicle_model.numpy_impl import VehicleModel
from gym_offroad_nav.vehicle_model.cython_impl import c_step as cython_step
import collections

@fixture
def datadir(tmpdir, request):
    '''
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    '''
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, bytes(tmpdir))

    return tmpdir

def to_contiguous(data):

    if isinstance(data, np.ndarray):
        data = np.ascontiguousarray(data)
    elif hasattr(data, 'iteritems'):
        for k, v in data.iteritems():
            data[k] = to_contiguous(v)
    elif isinstance(data, collections.Iterable):
        data = tuple([to_contiguous(x) for x in data])

    return data

def load_test_case(datadir, fn):
    f = open(str(datadir.join(fn)), 'rb')
    return to_contiguous(AttrDict(cPickle.load(f)))

class TestVehicleModel():

    def test_vehicle_model(self, datadir):

        for i in range(8):
            test_case = load_test_case(datadir, "test_case_%d.pkl" % i)

            outputs = cython_step(**test_case.inputs)

            for output, expected_output in zip(outputs, test_case.outputs):
                assert np.all(output == expected_output)
