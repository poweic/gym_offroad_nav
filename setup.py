#!/usr/bin/env python
#
# Enable cython support for eval scripts
# Run as
# setup.py build_ext --inplace
#
# WARNING: Only tested for Ubuntu 64bit OS.

try:
    from distutils.core import setup
    from distutils.extension import Extension
    from Cython.Build import cythonize
except:
    print "Unable to setup. Please use pip to install: cython"
    print "sudo pip install cython"
import os
import numpy

os.environ["CC"]  = "g++"
os.environ["CXX"] = "g++"

ext = Extension(
    "gym_offroad_nav.vehicle_model.cython_impl", ["gym_offroad_nav/vehicle_model/cython_impl.pyx"],
    include_dirs = [numpy.get_include()], #, '/usr/local/include/opencv', '/usr/local/include'],
    # libraries = ['opencv_core', 'opencv_highgui', 'opencv_imgproc'],
    extra_compile_args=['-Wno-cpp', '-std=c++11', '-O3'],
    language="c++"
)

setup(ext_modules = cythonize(ext))
