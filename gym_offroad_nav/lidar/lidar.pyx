# cython methods to speed-up evaluation

import numpy as np
cimport cython
cimport numpy as np
import ctypes

np.import_array()
from libc.stdint cimport uint8_t, int32_t, uint64_t, int64_t, uint32_t

cdef extern from "lidar_core.cpp":
    void mask(
        uint8_t* images, uint32_t batch_size, uint32_t height, uint32_t width,
        uint8_t threshold, uint32_t random_seed
    )

# @cython.boundscheck(False)
def c_lidar_mask(np.ndarray[np.uint8_t, ndim=4] images,
                 uint8_t threshold, uint32_t random_seed):

    cdef np.uint32_t batch_size = images.shape[0]
    cdef np.uint32_t height = images.shape[1]
    cdef np.uint32_t width = images.shape[2]

    # assume the last dimension is just 1
    mask(<uint8_t*>images.data, batch_size, height, width, threshold, random_seed)
