import pybnb


class RootInfeasibleMax(pybnb.Problem):

    #
    # Implement Problem abstract methods
    #

    def sense(self):
        return pybnb.maximize

    def objective(self):  # pragma:nocover
        raise NotImplementedError()

    def bound(self):
        return self.infeasible_objective()

    def save_state(self, node):
        pass

    def load_state(self, node):
        pass

    def branch(self):  # pragma:nocover
        raise NotImplementedError()
