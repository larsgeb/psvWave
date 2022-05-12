import os
import re
import sys
import platform
import subprocess
from sysconfig import get_paths, get_config_vars
import versioneer

from setuptools import setup, Extension, find_packages, Command
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


with open("README.md", "r") as fh:
    long_description = fh.read()

cmd_classes = versioneer.get_cmdclass()

setup(
    version=versioneer.get_version(),
    cmdclass=cmd_classes,
    name="psvWave",
    author="Lars Gebraad",
    author_email="lars.gebraad@erdw.ethz.ch",
    description="P-SV wave propagation in 2D for FWI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/larsgeb/forward-virieux",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "cmake",
        "pybind11",
        "matplotlib",
        "ipywidgets",
        "scipy",
    ],
    extras_require={
        "dev": [
            # Runtime
            "numpy",
            "matplotlib",
            # Build
            "pybind11",
            # Test
            "pytest",
            # Development
            "setuptools",
            "black",
            "flake8",
            "versioneer",
            # Documentation
            "sphinx",
            "sphinx_rtd_theme",
            "breathe",
            "m2r2",
            "htmlmin",
        ]
    },
    ext_modules=[
        Extension(
            name="__psvWave_cpp",
            sources=[
                "src/psvWave.cpp",
            ],
            include_dirs=["build"],
        )
    ],
    zip_safe=False,
)
