import math

from pybnb.common import (minimize,
                          maximize,
                          inf,
                          nan)

class Test(object):

    def test_minimize(self):
        assert minimize == 1

    def test_maximize(self):
        assert maximize == -1

    def test_inf(self):
        assert math.isinf(inf)
        assert math.isinf(-inf)

    def test_nan(self):
        assert math.isnan(nan)
        assert math.isnan(-nan)
