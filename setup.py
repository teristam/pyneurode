from setuptools import setup, Extension
import setuptools
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import subprocess
import sys


USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension(name='pyneurode.spike_sorter_cy', 
                  sources=['src/pyneurode/spike_sorter_cy'+ext], 
                  include_dirs=[numpy.get_include()],
                  language='c')]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    name='pyneurode',
    version='0.1.0',
    packages=setuptools.find_packages('src'),
    package_dir={'':'src'},
    author='Teris Tam',
    long_description=open('README.md').read(),
    ext_modules=extensions,
    install_requires=[
        'matplotlib',
        'dearpygui == 1.3.1',
        'scikit-learn',
        'seaborn',
        'numpy',
        'shortuuid',
        'networkx',
        'pyzmq',
        'jupyterlab',
        'scipy',
        'Cython',
        'Sphinx',
        'myst-parser',
        'palettable',
        'tqdm',
        'pytest',
        'sphinx_rtd_theme',
        'pyfirmata',
        'pyserial',
        'gitpython',
        'isosplit5 @ git+https://github.com/magland/isosplit5_python'
    ],
    zip_safe=False
)