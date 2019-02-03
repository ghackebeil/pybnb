import pybnb

class InfeasibleMin(pybnb.Problem):

    def __init__(self,
                 branching_abstol=0.001,
                 fixed_objective=None):
        if fixed_objective is None:
            fixed_objective = self.infeasible_objective()
        assert branching_abstol > 0
        self._branching_abstol = branching_abstol
        self._fixed_objective = fixed_objective
        self._xL = 0
        self._xU = 1

    #
    # Implement Problem abstract methods
    #

    def sense(self):
        return pybnb.minimize

    def objective(self):
        return self._fixed_objective

    def bound(self):
        delta = self._xU - self._xL
        if delta < 0.01:
            return self.infeasible_objective()
        else:
            return 1.0/delta

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
