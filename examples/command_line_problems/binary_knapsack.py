#
# This example solves the binary knapsack problem.
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

    def branch(self, node):
        assert len(self._choices) < self._n
        orig_state = node.state
        node.state = None
        for level in range(self._level, self._n):
            i = self._sorted_order[level]
            child_weight = self._weight + self._w[i]
            if child_weight <= self._W:
                child = node.new_child()
                child.state = (child_weight,
                               self._value + self._v[i],
                               level + 1,
                               self._choices + [i])
                yield child
        node.state = orig_state

if __name__ == "__main__":
    import pybnb.misc

    W = 25
    w = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
    v = [2.05,2.06,2.07,2.08,2.09,2.10,2.11,
         2.12,2.13,2.14,2.15,2.16,2.17,2.18,
         2.19,2.20,2.21,2.22,2.23,2.24]

    problem = BinaryKnapsack(W,v,w)
    pybnb.misc.create_command_line_solver(problem)
