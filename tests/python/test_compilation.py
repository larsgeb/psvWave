import subprocess
import glob
import os
from sysconfig import get_paths as gp
import pybind11


def test_compilation():

    PYTHON_INCLUDES = gp()["include"]
    PYBIND_INCLUDES = pybind11.get_include()

    out = subprocess.run(
        "echo $(python3-config --extension-suffix)",
        shell=True,
        capture_output=True,
        text=True,
    )
    SUFFIX = out.stdout.strip("\n")
    assert out.returncode == 0

    env = os.environ.copy()
    env["PYTHON_INCLUDES"] = PYTHON_INCLUDES
    env["PYBIND_INCLUDES"] = PYBIND_INCLUDES
    env["SUFFIX"] = SUFFIX

    out = subprocess.run("cmake .", shell=True, env=env)
    assert out.returncode == 0

    out = subprocess.run("make all", shell=True)
    assert out.returncode == 0


def test_cpp():

    for cpp_test in glob.glob("test_*"):
        out = subprocess.run(f"./{cpp_test}")
        assert out.returncode == 0
