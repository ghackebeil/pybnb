import math

import pybnb

try:
    from numba import jit
except ImportError:
    jit = lambda x: x

def log2floor(n):
    """
    Returns the exact value of floor(log2(n)).
    No floating point calculations are used.
    Requires positive integer type.
    """
    assert n > 0
    return n.bit_length() - 1

class Lipschitz1D(pybnb.Problem):

    _L = 12.0

    def __init__(self, xL, xU,  branch_abstol=1e-5):
        assert branch_abstol > 0
        assert xL <= xU
        self._branch_abstol = branch_abstol
        self._root_xL = xL
        self._root_xU = xU
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
        if node.size != 0:
            node.resize(0)
        # if the tree_id is None, then we know
        # this is the initial call to save state
        # at the root node
        if node.tree_id is None:
            node.tree_id = 0

    def load_state(self, node):
        # determine the problem state from the
        # node's tree id
        level = log2floor(node.tree_id+1)
        delta = (self._root_xU - self._root_xL)/float(2**level)
        level_index = node.tree_id - ((2**level) - 1)
        self._xL = self._root_xL + level_index*delta
        self._xU = self._xL + delta

    def branch(self, parent):
        L, U = self._xL, self._xU
        dist = float(U - L)
        if dist/2.0 < self._branch_abstol:
            return ()
        # branch and use the 0-based binary-heap labeling to
        # encode the state of the child nodes
        children = parent.new_children(2)
        assert parent.tree_id is not None
        children[0].tree_id = 2*parent.tree_id + 1
        children[1].tree_id = 2*parent.tree_id + 2
        return children

if __name__ == "__main__":
    import pybnb.misc

    problem = Lipschitz1D(0,10)
    pybnb.misc.create_command_line_solver(problem)
