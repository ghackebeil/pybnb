import pybnb

class DelayedUnboundedMax(pybnb.Problem):

    #
    # Implement Problem abstract methods
    #

    def sense(self):
        return pybnb.maximize
    def objective(self):
        if self.tree_id >= 10:
            return self.unbounded_objective()
        else:
            return 0
    def bound(self):
        return self.unbounded_objective()
    def save_state(self, node):
        pass
    def load_state(self, node):
        self.tree_id = node.tree_id
    def branch(self, node):
        yield node.new_child()
        yield node.new_child()
        yield node.new_child()
        yield node.new_child()
        yield node.new_child()
