"""
submission/setup.py -- builds the cython extensions.

run from inside submission/ (grading harness cds here first):
    cd submission && python setup.py build_ext --inplace

module names kept bare (_fast not submission._fast) so --inplace puts the
.so right next to the .pyx, which is where the import expects it.

both are pure speedups, everything falls back to pure python/numpy if
these don't compile. built at image-build time not in build_index() since
that would count against build-time efficiency
"""
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError:  # pragma: no cover - Cython is pinned in requirements.txt
    cythonize = None

import numpy as np

extensions = [
    Extension(
        "_fastbuild",
        sources=["_fastbuild.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++11"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "_fast",
        sources=["_fast.pyx"],
        include_dirs=[np.get_include()],
        # -ffp-contract=off matters -- without it the compiler fuses a*b+c
        # into one FMA which rounds differently than numpy, diverged on the
        # full corpus once even though a small test still passed
        extra_compile_args=["-O3", "-ffp-contract=off"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="a1-sparse-retrieval",
    ext_modules=cythonize(extensions, language_level=3) if cythonize else [],
    zip_safe=False,
)
