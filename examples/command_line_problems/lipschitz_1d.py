#
# This example solves a simple one dimensional problem that
# has an expensive objective function. A bound is computed
# using a specified lipschitz constant (assumed to be
# valid).
#
# The package numba is not required, but it will make
# this example run faster.
#
# Recommended usage:
#
# $ python lipschitz_1d.py --relative-gap=1e-4
#

import math

import pybnb

try:
    from numba import jit
except ImportError:
    jit = lambda x: x

class Lipschitz1D(pybnb.Problem):

    _LC = 12.0

    def __init__(self, xL, xU):
        assert xL <= xU
        self._xL = xL
        self._xM = 0.5*(xL+xU)
        self._xU = xU
        assert self._xL <= self._xM <= self._xU
        self._fL_cached = None
        self._fM_cached = None
        self._fU_cached = None

    @jit
    def _f(self, x):
        ans = 0
        for i in range(1000,0,-1):
            temp = 0
            for j in range(i,0,-1):
                temp += (x + j)**(-3.1)
            ans += math.sin(x + temp) / (1.2**i)
        return ans

    #
    # Implement Problem abstract methods
    #

    def sense(self):
        return pybnb.maximize

    def objective(self):
        if self._fM_cached is None:
            self._fM_cached = self._f(self._xM)
        return self._fM_cached

    def bound(self):
        if self._fL_cached is None:
            self._fL_cached = self._f(self._xL)
        if self._fU_cached is None:
            self._fU_cached = self._f(self._xU)
        return 0.5*self._fL_cached + \
               0.5*self._fU_cached + \
               0.5*self._LC*(self._xU-self._xL)

    def save_state(self, node):
        node.state = (self._xL,
                      self._xU,
                      self._fL_cached,
                      self._fU_cached)

    def load_state(self, node):
        (self._xL,
         self._xU,
         self._fL_cached,
         self._fU_cached) = node.state
        self._xM = 0.5*(self._xL+self._xU)
        self._fM_cached = None
        assert self._xL <= self._xM <= self._xU

    def branch(self):
        child = pybnb.Node()
        child.state = (self._xL,
                       self._xM,
                       self._fL_cached,
                       self._fM_cached)
        yield child
        child = pybnb.Node()
        child.state = (self._xM,
                       self._xU,
                       self._fM_cached,
                       self._fU_cached)
        yield child

if __name__ == "__main__":
    import pybnb.misc

    problem = Lipschitz1D(0,10)
    pybnb.misc.create_command_line_solver(problem)
