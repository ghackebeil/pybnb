import pybnb

class UnboundedMin(pybnb.Problem):

    def __init__(self):
        super(UnboundedMin, self).\
            __init__(pybnb.minimize)

    #
    # Implement Problem abstract methods
    #

    def objective(self): return self.unbounded_objective
    def bound(self): return self.unbounded_objective
    def save_state(self, node): pass
    def load_state(self, node): pass
    def branch(self): raise NotImplementedError()
