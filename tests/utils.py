import os
import cPickle
import numpy as np
from attrdict import AttrDict
from distutils import dir_util
from pytest import fixture
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

def run_function_with_test_case(func, test_case):

    outputs = func(**test_case.inputs)

    def compare(A, B):
        for a, b in zip(A, B):
            assert np.all(a == b)

    # if the function return a output, then compare the outputs
    if outputs is not None:
        compare(outputs, test_case.outputs)

    # if the function changes its inputs, then compare inputs before/after exec
    if hasattr(test_case, 'inputs_after_exec'):
        compare(test_case.inputs, test_case.inputs_after_exec)
