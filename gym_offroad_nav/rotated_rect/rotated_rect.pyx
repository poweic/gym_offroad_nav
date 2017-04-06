# cython methods to speed-up evaluation

import numpy as np
cimport cython
cimport numpy as np
import ctypes

np.import_array()
from libc.stdint cimport uint8_t, int32_t, uint64_t, int64_t, uint32_t

cdef extern from "rotated_rect_impl.cpp":
    void get_rotated_rect(
        uint8_t* input,
        uint32_t height,
        uint32_t width,
        uint32_t channels,
        uint8_t* output,
        uint32_t cx,
        uint32_t cy,
        uint32_t sizex,
        uint32_t sizey,
        float angle)

@cython.boundscheck(False)
def c_rotated_rect(np.ndarray[np.uint8_t, ndim=3] input_np,
                   tuple center, tuple size, float angle):

    cdef np.uint32_t width = input_np.shape[0]
    cdef np.uint32_t height = input_np.shape[1]
    cdef np.uint32_t channels = input_np.shape[2]
    
    cdef np.ndarray[np.uint8_t, ndim=3, mode="c"] input_c
    cdef np.ndarray[np.uint8_t, ndim=3, mode="c"] output_c
    
    input_c = np.ascontiguousarray(input_np, dtype=np.uint8)
    output_c = np.ascontiguousarray(np.zeros(
        (size[0], size[1], channels), dtype=np.uint8))

    get_rotated_rect(
        &input_c[0, 0, 0],
        height, width, channels,
        &output_c[0, 0, 0],
        center[0], center[1],
        size[1], size[0],
        angle
    )

    return output_c
