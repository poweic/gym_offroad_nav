# cython methods to speed-up evaluation

import numpy as np
cimport cython
cimport numpy as np
import ctypes

np.import_array()
from libc.stdint cimport uint8_t, int32_t, uint64_t, int64_t, uint32_t

cdef extern from "vehicle_model.cpp":
    void step(
        double* x, double* s, const double* u, const double* n,
        uint32_t n_sub_steps, uint32_t batch_size,
        float dt, float noise_level, float wheelbase, float drift
    )

@cython.boundscheck(False)
def c_step(np.ndarray[np.float64_t, ndim=2] x,
           np.ndarray[np.float64_t, ndim=2] y,
           np.ndarray[np.float64_t, ndim=2] u,
           np.ndarray[np.float64_t, ndim=3] n,
           uint32_t n_sub_steps, uint32_t batch_size,
           float timestep, float noise_level, float wheelbase, float drift):

    step(
        <double*>x.data,
        <double*>y.data,
        <double*>u.data,
        <double*>n.data,
        n_sub_steps, batch_size,
        timestep, noise_level, wheelbase, drift
    )
