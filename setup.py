from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "tensorpack._tensorpack_csrc",
        [
            "csrc/src/nvml_fast_wrapper.cpp",
            "csrc/pybind_bindings.cpp",
        ],
        include_dirs=["csrc/include"],
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(where="python"),
    package_dir={"": "python"},
)
