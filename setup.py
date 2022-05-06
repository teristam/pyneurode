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
    include_dirs=[numpy.get_include()], #for cythons
    setup_requries=[
        'cython',
        'setuptools>=18.0',
    ],
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
        'scipy[all]',
        'Cython',
        'Sphinx',
        'myst-parser',
        'palettable',
        'tqdm',
    ]
)