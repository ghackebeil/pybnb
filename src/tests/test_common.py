from pybnb.common import (minimize,
                          maximize,
                          infinity,
                          is_infinite)

import numpy

class Test(object):

    def test_minimize(self):
        assert minimize == 1

    def test_maximize(self):
        assert maximize == -1

    def test_is_infinite(self):
        assert infinity == float('inf')
        assert -infinity == float('-inf')
        assert is_infinite(infinity)
        assert is_infinite(-infinity)
        assert is_infinite(float('inf'))
        assert is_infinite(float('-inf'))
        assert is_infinite(numpy.inf)
        assert is_infinite(-numpy.inf)

        assert not is_infinite(0.0)
        assert not is_infinite(numpy.nan)
        assert not is_infinite(infinity - infinity)
