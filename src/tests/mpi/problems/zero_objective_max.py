import pybnb

class ZeroObjectiveMax(pybnb.Problem):

    def __init__(self,
                 branching_abstol=0.001):
        assert branching_abstol > 0
        self._branching_abstol = branching_abstol
        self._xL = 0
        self._xU = 1

    #
    # Implement Problem abstract methods
    #

    def sense(self):
        return pybnb.maximize

    def objective(self):
        return 0.0

    def bound(self):
        return self._xU - self._xL

    def save_state(self, node):
        node.state = (self._xL, self._xU)

    def load_state(self, node):
        (self._xL, self._xU) = node.state

    def branch(self):
        xL, xU = self._xL, self._xU
        if (xU - xL) <= self._branching_abstol:
            return
        mid = 0.5 * (xL + xU)
        child = pybnb.Node()
        child.state = (xL, mid)
        yield child
        child = pybnb.Node()
        child.state = (mid, xU)
        yield child
