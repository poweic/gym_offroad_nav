# cython methods to speed-up evaluation

import numpy as np
cimport cython
cimport numpy as np
import ctypes

np.import_array()
from libc.stdint cimport uint8_t, int32_t, uint64_t, int64_t, uint32_t

cdef extern from "vehicle_model.cpp":
    void step(
        # For vehicle Ax + Bu and Cx + Du
        double* x, double* s, const double* u, const double* n,
        uint32_t n_sub_steps, uint32_t batch_size,
        float dt, float noise_level, float wheelbase, float drift,
        # For collecting rewards during the simulation
        float* rewards, const float* const reward_map,
        uint32_t height, uint32_t width,
        int32_t x_min, int32_t x_max,
        int32_t y_min, int32_t y_max,
        float cell_size
    )

@cython.boundscheck(False)
def c_step(np.ndarray[np.float64_t, ndim=2] x,
           np.ndarray[np.float64_t, ndim=2] y,
           np.ndarray[np.float64_t, ndim=2] u,
           np.ndarray[np.float64_t, ndim=3] n,
           uint32_t n_sub_steps, uint32_t batch_size,
           float timestep, float noise_level, float wheelbase, float drift,
           # For collecting rewards during the simulation
           np.ndarray[np.float32_t, ndim=2] rewards,
           np.ndarray[np.float32_t, ndim=2] reward_map,
           int32_t x_min, int32_t x_max,
           int32_t y_min, int32_t y_max,
           float cell_size):

    cdef np.uint32_t height = reward_map.shape[0]
    cdef np.uint32_t width = reward_map.shape[1]

    step(
        <double*>x.data,
        <double*>y.data,
        <double*>u.data,
        <double*>n.data,
        n_sub_steps, batch_size,
        timestep, noise_level, wheelbase, drift,
        <float*> rewards.data, <float*> reward_map.data, height, width,
        x_min, x_max, y_min, y_max, cell_size
    )
