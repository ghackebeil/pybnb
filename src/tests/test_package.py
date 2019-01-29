import sys

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
                    pass
            elif sys.version_info.major == 3:
                if sys.version_info.minor == 5:
                    pass
                elif sys.version_info.minor == 6:
                    pass
                elif sys.version_info.minor == 7:
                    pass
        if is_pypy:
            if sys.version_info.major == 2:
                pass
            if sys.version_info.major == 3:
                pass

    def test_version(self):
        pybnb.__version__
