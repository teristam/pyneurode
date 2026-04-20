from setuptools import setup, Extension
import numpy

USE_CYTHON = True

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension(
    name='pyneurode.spike_sorter_cy',
    sources=['src/pyneurode/spike_sorter_cy' + ext],
    include_dirs=[numpy.get_include()],
    language='c',
)]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(ext_modules=extensions)
