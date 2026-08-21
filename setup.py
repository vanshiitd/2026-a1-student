"""
setup.py — builds the optional C extension.

The extension in submission/_fast.pyx fuses VByte decoding with BM25 scoring;
profiling put 90% of query time in the phases it replaces. It is strictly an
optimisation: every caller imports it behind try/except and falls back to a
pure-Python path, so a submission where this fails to build still runs correctly
(just slower).

Built at image-build time, never inside build_index() -- anything build_index()
does is charged against the index-build-time efficiency metric, and a one-time
compile is not indexing work.

    python setup.py build_ext --inplace
"""
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError:  # pragma: no cover - Cython is pinned in requirements.txt
    cythonize = None

import numpy as np

extensions = [
    Extension(
        "submission._fast",
        sources=["submission/_fast.pyx"],
        include_dirs=[np.get_include()],
        # -O3 for speed, but two float-safety flags are mandatory:
        #   -ffast-math      would permit reassociation of float operations.
        #   -ffp-contract=off stops the compiler fusing `a*b + c` into a single
        #                    FMA instruction, which rounds ONCE instead of twice
        #                    and so produces different (not wrong, but different)
        #                    results from NumPy.
        # Without the second flag the kernel diverged from the NumPy path on the
        # full corpus while still passing on a small fixture -- caught only by
        # comparing full-corpus rankings.
        extra_compile_args=["-O3", "-ffp-contract=off"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="a1-sparse-retrieval",
    ext_modules=cythonize(extensions, language_level=3) if cythonize else [],
    zip_safe=False,
)
