#
# This example solves the binary knapsack problem.
#
# Recommended usage:
#
# $ python binary_knapsack.py
#

import pybnb

class BinaryKnapsack(pybnb.Problem):

    def __init__(self, W, v, w):
        assert W >= 0
        assert len(v) == len(w)
        self._W = W
        self._v = list(v)
        self._w = list(w)
        self._n = len(self._v)
        self._sorted_order = sorted(
            range(self._n),
            key=lambda i: self._v[i]/float(self._w[i]),
            reverse=True)
        self._weight = 0
        self._value = 0
        self._level = 0
        self._choices = []

    #
    # Implement Problem abstract methods
    #

    def sense(self):
        return pybnb.maximize

    def objective(self):
        return self._value

    def bound(self):
        weight = self._weight
        bound = self._value
        added = False
        for level in range(self._level, self._n):
            k = self._sorted_order[level]
            next_weight = weight + self._w[k]
            if next_weight > self._W:
                break
            weight = next_weight
            bound += self._v[k]
            added = True
        else: # no break in for-loop
            # there are no items left
            # to add partially
            return bound
        if not added:
            # terminal node
            return self.objective()
        return bound + \
            (self._W - weight)*(self._v[k]/float(self._w[k]))

    def save_state(self, node):
        node.state = (self._weight,
                      self._value,
                      self._level,
                      self._choices)

    def load_state(self, node):
        (self._weight,
         self._value,
         self._level,
         self._choices) = node.state
        assert len(self._choices) <= self._n
        assert self._weight <= self._W
        assert self._level <= self._n

    def branch(self):
        assert len(self._choices) < self._n
        for level in range(self._level, self._n):
            i = self._sorted_order[level]
            child_weight = self._weight + self._w[i]
            if child_weight <= self._W:
                child_value = self._value + self._v[i]
                child = pybnb.Node()
                # we know the child objective value, so
                # assign it to the child node rather than
                # letting it inherit the parent objective
                # value (this may be useful for queue
                # prioritization)
                child.objective = child_value
                child.state = (child_weight,
                               child_value,
                               level + 1,
                               self._choices + [i])
                yield child

if __name__ == "__main__":
    import pybnb.misc

    W = 25
    w = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
    v = [205,206,207,208,209,210,211,
         212,213,214,215,216,217,218,
         219,220,221,222,223,224]

    problem = BinaryKnapsack(W,v,w)
    pybnb.misc.create_command_line_solver(problem)
