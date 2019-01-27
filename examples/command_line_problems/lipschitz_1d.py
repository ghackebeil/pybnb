#
# This example solves a simple one dimensional problem that
# has an expensive objective function. A bound is computed
# using a specified lipschitz constant (assumed to be
# valid).
#
# The package numba is not required, but it will make
# this example run faster.
#

import math

import pybnb

try:
    from numba import jit
except ImportError:
    jit = lambda x: x

class Lipschitz1D(pybnb.Problem):

    _LC = 12.0

    def __init__(self, xL, xU,  branch_abstol=1e-5):
        assert branch_abstol > 0
        assert xL <= xU
        self._branch_abstol = branch_abstol
        self._xL = xL
        self._xU = xU
        self._fL_cached = self._f(self._xL)
        self._fU_cached = self._f(self._xU)
        self._fmid_cached = self._f(0.5*(self._xL + self._xU))

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
        if self._fmid_cached is None:
            mid = 0.5 * (self._xL + self._xU)
            self._fmid_cached = self._f(mid)
        return self._fmid_cached

    def bound(self):
        return 0.5*self._fL_cached + \
               0.5*self._fU_cached + \
               0.5*self._LC*(self._xU-self._xL)

    def save_state(self, node):
        node.state = (self._xL,
                      self._xU,
                      self._fL_cached,
                      self._fU_cached,
                      self._fmid_cached)

    def load_state(self, node):
        (self._xL,
         self._xU,
         self._fL_cached,
         self._fU_cached,
         self._fmid_cached) = node.state

    def branch(self, node):
        dist = float(self._xU - self._xL)
        if 0.5*dist < self._branch_abstol:
            return ()

        # save the current state in the argument node, so we
        # can easily reset
        self.save_state(node)

        # branch
        xL, xU = self._xL, self._xU
        fL, fU = self._fL_cached, self._fU_cached
        fmid = self._fmid_cached
        children = [node.new_child() for i in range(2)]
        mid = 0.5*(xL + xU)

        # left child
        self._xL = xL
        self._xU = mid
        self._fL_cached = fL
        self._fU_cached = fmid
        self._fmid_cached = None
        self.save_state(children[0])

        # right child
        self._xL = mid
        self._xU = xU
        self._fL_cached = fmid
        self._fU_cached = fU
        self._fmid_cached = None
        self.save_state(children[1])

        # reset the current state
        self.load_state(node)

        return children

if __name__ == "__main__":
    import pybnb.misc

    problem = Lipschitz1D(0,10)
    pybnb.misc.create_command_line_solver(problem)
