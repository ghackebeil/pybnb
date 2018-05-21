import pybnb

class InfeasibleMax(pybnb.Problem):

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
        return pybnb.maximize

    def objective(self):
        return self._fixed_objective

    def bound(self):
        delta = self._xU - self._xL
        if delta < 0.01:
            return self.infeasible_objective()
        else:
            return -1.0/delta

    def save_state(self, node):
        node.resize(2)
        state = node.state
        state[0] = self._xL
        state[1] = self._xU

    def load_state(self, node):
        state = node.state
        self._xL = float(state[0])
        self._xU = float(state[1])

    def branch(self, parent):
        xL, xU = self._xL, self._xU
        if (xU - xL) <= self._branching_abstol:
            return ()
        children = [parent.new_child()
                    for i in range(2)]
        mid = 0.5 * (xL + xU)
        self._xL = xL
        self._xU = mid
        self.save_state(children[0])
        self._xL = mid
        self._xU = xU
        self.save_state(children[1])
        self._xL, self._xU = xL, xU
        return children
