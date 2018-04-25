import sys

import pytest

import pybnb

is_pypy = False
try:
    import __pypy__
    is_pypy = True
except ImportError:
    is_pypy = False

class Test(object):

    # See what Python versions the combined
    # coverage report includes
    def test_show_coverage(self):
        if not is_pypy:
            if sys.version_info.major == 2:
                if sys.version_info.minor == 7:
                    print(sys.version_info)
            elif sys.version_info.major == 3:
                if sys.version_info.minor == 5:
                    print(sys.version_info)
                elif sys.version_info.minor == 6:
                    print(sys.version_info)
        if is_pypy:
            if sys.version_info.major == 2:
                print(sys.version_info)
            if sys.version_info.major == 3:
                print(sys.version_info)

    def test_version(self):
        pybnb.__version__
