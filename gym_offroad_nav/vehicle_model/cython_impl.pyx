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
        double* x, double* s, const double* u,
        uint32_t n_sub_steps, uint32_t batch_size,
        float dt, float noise_level, float wheelbase, float drift,
        uint32_t random_seed,
        # For collecting rewards during the simulation
        float* rewards, const float* const reward_map,
        uint32_t height, uint32_t width,
        int32_t x_min, int32_t x_max,
        int32_t y_min, int32_t y_max,
        float cell_size,
        float low_speed_penalty, float decay_rate, float high_acc_penalty
    )

@cython.boundscheck(False)
def c_step(np.ndarray[np.float64_t, ndim=2] x,
           np.ndarray[np.float64_t, ndim=2] s,
           np.ndarray[np.float64_t, ndim=2] u,
           uint32_t n_sub_steps,
           float timestep, float noise_level, float wheelbase, float drift,
           uint32_t random_seed,
           # For collecting rewards during the simulation
           np.ndarray[np.float32_t, ndim=2] reward_map,
           dict bounds, float cell_size,
           float low_speed_penalty, float decay_rate, float high_acc_penalty):

    # state s comes in column-vector (i.e. 6 x batch_size)
    cdef np.uint32_t batch_size = s.shape[1]

    # get height and width of the reward_map
    cdef np.uint32_t height = reward_map.shape[0]
    cdef np.uint32_t width = reward_map.shape[1]

    # allocate contiguous C-order rewards
    cdef np.ndarray rewards = np.zeros((1, batch_size), dtype=np.float32, order='C')

    step(
        <double*>x.data,
        <double*>s.data,
        <double*>u.data,
        n_sub_steps, batch_size,
        timestep, noise_level, wheelbase, drift,
        random_seed,
        <float*> rewards.data, <float*> reward_map.data, height, width,
        bounds["x_min"], bounds["x_max"], bounds["y_min"], bounds["y_max"], cell_size,
        low_speed_penalty, decay_rate, high_acc_penalty
    )

    return rewards
