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


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system(
            "rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info ./psvWave/*.so "
            "__pycache__/"
        )


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):

        try:
            out = subprocess.check_output(["cmake", "--version"])
            print(out)
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):

        import pybind11

        extension = get_config_vars()["EXT_SUFFIX"]
        python_includes = get_paths()["include"]
        pybind_includes = pybind11.get_include()

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DPYBIND_INCLUDES=" + pybind_includes,
            "-DPYTHON_INCLUDES=" + python_includes,
            "-DEXTENSION=" + extension,
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += [
                "-DCMAKE_BUILD_TYPE=" + cfg,
                "-O3",
            ]
            build_args += ["--", "-j2"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        out = subprocess.Popen(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=self.build_temp,
            env=env,
            stdout=subprocess.PIPE,
        )

        result = out.communicate()
        print(result)

        out = subprocess.Popen(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

        result = out.communicate()
        print(result)


with open("README.md", "r") as fh:
    long_description = fh.read()

cmd_classes = versioneer.get_cmdclass()
cmd_classes["build_ext"] = CMakeBuild
cmd_classes["clean"] = CleanCommand

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
    install_requires=["numpy", "cmake", "pybind11", "matplotlib"],
    extras_require={
        "dev": [
            "numpy",
            "cmake",
            "pybind11",
            "matplotlib",
            "black",
            "setuptools",
            "matplotlib",
            "versioneer",
            "sphinx",
            "sphinx_rtd_theme",
            "breathe",
            "m2r @ git+https://github.com/crossnox/m2r@dev#egg=m2r",
        ]
    },
    ext_modules=[CMakeExtension("__psvWave_cpp", ".")],
    zip_safe=False,
)
