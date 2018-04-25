import pybnb

class UnboundedMax(pybnb.Problem):

    def __init__(self):
        super(UnboundedMax, self).\
            __init__(pybnb.maximize)

    #
    # Implement Problem abstract methods
    #

    def objective(self): return self.unbounded_objective
    def bound(self): return self.unbounded_objective
    def save_state(self, node): pass
    def load_state(self, node): pass
    def branch(self): raise NotImplementedError()
