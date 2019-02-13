#
# This example defines a script that solves the Traveling
# Salesperson Problem using a combination of
# branch-and-bound (by edge) and local heuristics. It
# highlights a number of advanced pybnb features, including:
#  (1) the use of pybnb.futures.NestedSolver
#  (2) re-continuing a solve after early termination
#
# This example can be executed in serial as
#
# $ python tsp_byedge.py <data_file>
#
# or in parallel as
#
# $ mpiexec -n <n> python tsp_byedge.py <data_file>
#
# The following data files are available:
# (source: https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html)
#  - p01_d.txt: 15 cities, minimal tour length 291
#  - p01_d_inf.txt: same as above, but with random paths
#                   removed to make the problem infeasible
#  - fri26_d.txt: 26 cities, minimal tour length 937
#
import pybnb

try:
    import numpy
except ImportError:                               #pragma:nocover
    raise ImportError("This example requires numpy")

class TSP_ByEdge(pybnb.Problem):

    def __init__(self, dist):
        self._N = len(dist)
        # state that changes during the solve
        self._dist = numpy.array(dist,
                                 dtype=float)
        numpy.fill_diagonal(self._dist,
                            numpy.inf)
        self._path = [0]
        self._partial_cost = 0
        self._cost = None

    def _row_reduction(self):
        row_mins = self._dist.min(axis=1)
        mask = (row_mins != numpy.inf)
        tmp = row_mins[mask, numpy.newaxis]
        self._dist[mask, :] -= tmp
        return tmp.sum()

    def _col_reduction(self):
        col_mins = self._dist.min(axis=0)
        mask = (col_mins != numpy.inf)
        tmp = col_mins[mask]
        self._dist[:, mask] -= tmp
        return tmp.sum()

    #
    # Implement Problem abstract methods
    #

    def sense(self):
        return pybnb.minimize

    def objective(self):
        if len(self._path) == self._N:
            assert self._cost is not None
            return self._cost
        else:
            return self.infeasible_objective()

    def bound(self):
        if self._cost is None:
            assert len(self._path) >= 1
            if len(self._path) > 1:
                u = self._path[-2]
                v = self._path[-1]
                self._dist[u,:] = numpy.inf
                self._dist[:,v] = numpy.inf
                self._dist[v][self._path[0]] = numpy.inf
            row_sum = self._row_reduction()
            col_sum = self._col_reduction()
            self._cost = self._partial_cost
            self._cost += row_sum
            self._cost += col_sum
        return self._cost

    def save_state(self, node):
        node.state = (self._path,
                      self._dist,
                      self._partial_cost,
                      self._cost)

    def load_state(self, node):
        (self._path,
         self._dist,
         self._partial_cost,
         self._cost) = node.state
        assert len(self._path) <= self._N

    def branch(self):
        # note that the branch method should never be called
        # with a path of length N as the objective and bound
        # converge exactly in that case.
        assert len(self._path) < self._N
        assert self._cost is not None
        u = self._path[-1]
        candidates = numpy.flatnonzero(
            self._dist[u,:] != numpy.inf).tolist()
        if len(candidates) == 0:
            # this path is infeasible, so return a dummy
            # child to indicate that
            child = pybnb.Node()
            child.bound = pybnb.inf
            child.objective = pybnb.inf
            yield child
        else:
            for v in candidates:
                child = pybnb.Node()
                child.state = (self._path + [v],
                               self._dist.copy(),
                               self._cost + self._dist[u][v],
                               None)
                yield child

    def notify_solve_finished(self,
                              comm,
                              worker_comm,
                              results):
        tour = None
        if (results.best_node is not None) and \
           (results.best_node.state is not None):
            path = results.best_node.state[0]
            route = path + [path[0]]
            tour = {'cost': results.best_node.objective,
                    'route': route}
        results.tour = tour

if __name__ == "__main__":
    import argparse

    from tsp_util import (parse_dense_distance_matrix,
                          run_solve_loop)

    parser = argparse.ArgumentParser(
        description=("Run parallel branch and bound "
                     "to solve an instance of TSP."))
    parser.add_argument("data_filename", type=str,
                        help=("The name of a file that stores a "
                              "dense distance matrix."))
    parser.add_argument("--results-filename", type=str, default=None,
                        help=("When set, saves the solver results "
                              "into a YAML-formatted file with the "
                              "given name."))
    args = parser.parse_args()

    dist = parse_dense_distance_matrix(args.data_filename)
    problem = TSP_ByEdge(dist)

    solver = pybnb.Solver()
    results = run_solve_loop(dist, problem, solver)
    stats = solver.collect_worker_statistics()
    if solver.is_dispatcher:
        pybnb.solver.summarize_worker_statistics(stats)
        # save results to a file
        # (mainly used for testing this example)
        if args.results_filename is not None:
            results.write(args.results_filename)
