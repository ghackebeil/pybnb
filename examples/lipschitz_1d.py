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
        mid = 0.5 * (self._xL + self._xU)
        return self._f(mid)

    def bound(self):
        xL, xU, L, f = self._xL, self._xU, self._L, self._f
        return 0.5*f(xL) + 0.5*f(xU) + 0.5*L*(xU-xL)

    def save_state(self, node):
        node.resize(5)
        state = node.state
        state[0] = self._xL
        state[1] = self._xU

    def load_state(self, node):
        state = node.state
        self._xL = float(state[0])
        self._xU = float(state[1])

    def branch(self, parent):
        L, U = self._xL, self._xU
        dist = float(U - L)
        if dist/2.0 < self._branch_abstol:
            return ()
        # branch
        children = parent.new_children(2)
        mid = 0.5*(L + U)
        self._xL, self._xU = L, mid
        self.save_state(children[0])
        self._xL, self._xU = mid, U
        self.save_state(children[1])
        # reset the bounds
        self._xL, self._xU = L, U
        return children

if __name__ == "__main__":
    import pybnb.misc

    problem = Lipschitz1D(0,10)
    pybnb.misc.create_command_line_solver(problem)
