import pybnb

class DelayedUnboundedMax(pybnb.Problem):

    #
    # Implement Problem abstract methods
    #

    def sense(self):
        return pybnb.maximize
    def objective(self):
        if self._node.tree_depth > 1:
            return self.unbounded_objective()
        else:
            return 0
    def bound(self):
        return self.unbounded_objective()
    def save_state(self, node):
        pass
    def load_state(self, node):
        self._node = node
    def branch(self):
        yield pybnb.Node()
        yield pybnb.Node()
        yield pybnb.Node()
        yield pybnb.Node()
        yield pybnb.Node()
