from setuptools import setup, Extension
import pybind11
import os

# Project root = parent of rl/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ext = Extension(
    'quoridor_cpp',
    sources=['quoridor_bind.cpp'],
    include_dirs=[
        pybind11.get_include(),
        project_root,  # for quoridor.hpp
    ],
    language='c++',
    extra_compile_args=['-std=c++17', '-O2', '-fPIC'],
)

setup(
    name='quoridor_cpp',
    version='1.0',
    description='Quoridor C++ game logic - pybind11 bindings',
    ext_modules=[ext],
    python_requires='>=3.8',
)
