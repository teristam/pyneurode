from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name='pyneurode',
    version='0.1.0',
    packages=['pyneurode',],
    author='Teris Tam',
    long_description=open('README.md').read(),
    ext_modules=cythonize('pyneurode/spike_sorter_cy.pyx'),
    include_dirs=[numpy.get_include()]
)