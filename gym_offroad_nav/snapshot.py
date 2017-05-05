import scipy.io
import inspect
import itertools
from collections import OrderedDict
import cPickle
from copy import deepcopy

def memory_snapshot_decorate(filename):

    def memory_snapshot_decorate(func):

        def wrapper(*args, **kwargs):

            args_name = list(OrderedDict.fromkeys(inspect.getargspec(func)[0] + kwargs.keys()))
            inputs = OrderedDict(list(itertools.izip(args_name, args)) + list(kwargs.iteritems()))
            inputs_copy = deepcopy(inputs)

            outputs = func(*args, **kwargs)

            cPickle.dump(dict(inputs=inputs_copy, inputs_after_exec=inputs, outputs=outputs),
                         open(filename, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

            return outputs

        return wrapper

    return memory_snapshot_decorate
