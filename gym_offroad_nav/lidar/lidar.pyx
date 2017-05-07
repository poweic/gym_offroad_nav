# cython methods to speed-up evaluation

import numpy as np
cimport cython
cimport numpy as np
import ctypes

np.import_array()
from libc.stdint cimport uint8_t, int32_t, uint64_t, int64_t, uint32_t

cdef extern from "lidar_core.cpp":
    void mask(
        uint8_t* images,
        uint8_t* reward_map,
        const int32_t* const centers,
        const float* const angle,
        const int32_t& pivot_x, const int32_t& pivot_y, const float& scale,
        const uint32_t& batch_size, const uint32_t& height, const uint32_t& width,
        const uint32_t& in_height, const uint32_t& in_width,
        const double* const obj_positions, const uint8_t* const valids, uint32_t n_obj,
        const double* const states, float cell_size, uint32_t radius,
        const uint8_t& threshold, const uint32_t& random_seed
    )

# @cython.boundscheck(False)
def c_lidar_mask(
        np.ndarray[np.uint8_t, ndim=3] reward_map,
        np.ndarray[np.int32_t, ndim=2] centers,
        np.ndarray[np.float32_t, ndim=1] angle, tuple pivot, float scale,
        np.ndarray[np.uint8_t, ndim=4] images,
        np.ndarray[np.float64_t, ndim=2] obj_positions,
        np.ndarray[np.uint8_t, ndim=2] valids,
        np.ndarray[np.float64_t, ndim=2] states,
        float cell_size, uint32_t radius, uint8_t threshold, uint32_t random_seed):

    cdef np.uint32_t batch_size = images.shape[0]
    cdef np.uint32_t height = images.shape[1]
    cdef np.uint32_t width = images.shape[2]

    cdef np.uint32_t in_height = reward_map.shape[0]
    cdef np.uint32_t in_width = reward_map.shape[1]

    cdef np.uint32_t n_obj = obj_positions.shape[0]

    # assume the last dimension is just 1
    mask(
        <uint8_t*> images.data,
        <uint8_t*> reward_map.data,
        <int32_t*> centers.data,
        <float*> angle.data, pivot[0], pivot[1], scale,
        batch_size, height, width, in_height, in_width,
        <double*> obj_positions.data, <uint8_t*> valids.data, n_obj,
        <double*> states.data, cell_size, radius,
        threshold, random_seed
    )
