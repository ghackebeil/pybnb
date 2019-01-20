#
# This example script defines and solves
# a minimal pybnb example problem.
#
# It can be executed in serial as
#
# $ python simple.py
#
# or in parallel as
#
# $ mpiexec -n <n> python simple.py
#
# The mpi4py module is required.
#
import pybnb

class Simple(pybnb.Problem):
    def __init__(self):
        self._xL, self._xU = 0, 1
    #
    # required methods
    #
    def sense(self):
        return pybnb.minimize
    def objective(self):
        return round(self._xU-self._xL,3)
    def bound(self):
        return -(self._xU - self._xL)**2
    def save_state(self, node):
        node.resize(2)
        node.state[0] = self._xL
        node.state[1] = self._xU
    def load_state(self, node):
        self._xL = float(node.state[0])
        self._xU = float(node.state[1])
    def branch(self, node):
        xL, xU = self._xL, self._xU
        xM = 0.5 * (xL + xU)
        self._xL, self._xU = xL, xM
        left = node.new_child()
        self.save_state(left)
        self._xL, self._xU = xM, xU
        right = node.new_child()
        self.save_state(right)
        self._xL, self._xU = xL, xU
        return left, right
    #
    # optional methods
    #
    def notify_solve_begins(self,
                            comm,
                            worker_comm,
                            convergence_checker):
        pass
    def notify_new_best_objective_received(self,
                                           objective):
        pass
    def notify_new_best_objective(self,
                                  objective):
        pass
    def notify_solve_finished(self,
                              comm,
                              worker_comm,
                              results):
        pass

problem = Simple()
solver = pybnb.Solver()
results = solver.solve(problem)
