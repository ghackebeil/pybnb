import pybnb

class RootInfeasibleMin(pybnb.Problem):

    def __init__(self):
        super(RootInfeasibleMin, self).\
            __init__(pybnb.minimize)

    #
    # Implement Problem abstract methods
    #

    def objective(self): return self.infeasible_objective
    def bound(self): return self.infeasible_objective
    def save_state(self, node): pass
    def load_state(self, node): pass
    def branch(self): raise NotImplementedError()
