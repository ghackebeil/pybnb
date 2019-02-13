#
# This example defines a script that solves the Traveling
# Salesperson Problem using a combination of
# branch-and-bound (by vertex) and local heuristics. It
# highlights a number of advanced pybnb features, including:
#  (1) the use of pybnb.futures.NestedSolver
#  (2) re-continuing a solve after early termination
#
# This example can be executed in serial as
#
# $ python tsp_byvertex.py <data_file>
#
# or in parallel as
#
# $ mpiexec -n <n> python tsp_byvertex.py <data_file>
#
# The following data files are available:
# (source: https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html)
#  - p01_d.txt: 15 cities, minimal tour length 291
#  - p01_d_inf.txt: same as above, but with random paths
#                   removed to make the problem infeasible
#  - fri26_d.txt: 26 cities, minimal tour length 937
#
import pybnb

class TSP_ByVertex(pybnb.Problem):

    def __init__(self, dist):
        self._dist = dist
        self._N = len(dist)
        # state that changes during the solve
        self._path = [0]

    #
    # Implement Problem abstract methods
    #

    def sense(self):
        return pybnb.minimize

    def objective(self):
        cost = self.infeasible_objective()
        if len(self._path) == self._N:
            return_cost = self._dist[self._path[-1]][self._path[0]]
            assert return_cost != 0
            if return_cost != pybnb.inf:
                cost = 0.0
                for i in range(self._N-1):
                    cost += self._dist[self._path[i]][self._path[i+1]]
                cost += return_cost
        return cost

    def bound(self):
        if len(self._path) == self._N:
            return self.objective()

        visited = set(self._path)
        remaining = []
        for u in range(self._N):
            if u not in visited:
                remaining.append(u)
        assert len(remaining) > 0

        bound = 0
        # for the edges that are certain
        for i in range(len(self._path) - 1):
            bound += self._dist[self._path[i]][self._path[i+1]]
        # for the last item
        last = self._path[-1]
        tmp = [self._dist[last][v] for v in remaining
               if ((self._dist[last][v] != pybnb.inf) and \
                   (v != last))]
        if len(tmp) == 0:
            return self.infeasible_objective()
        bound += min(tmp)
        # for the undetermined nodes
        p = [self._path[0]] + remaining
        for r in remaining:
            tmp = [self._dist[r][v] for v in p
                   if ((self._dist[r][v] != pybnb.inf) and \
                       (v != r))]
            if len(tmp) == 0:
                return self.infeasible_objective()
            bound += min(tmp)
        return bound

    def save_state(self, node):
        node.state = (self._path,)

    def load_state(self, node):
        assert len(node.state) == 1
        self._path = node.state[0]
        assert len(self._path) <= self._N

    def branch(self):
        # note that the branch method should never be called
        # with a path of length N as the objective and bound
        # converge exactly in that case.
        assert len(self._path) < self._N
        u = self._path[-1]
        visited = set(self._path)
        for v in range(self._N):
            # dist[u][v] == inf means no edge
            if (self._dist[u][v] != pybnb.inf) and \
               (v not in visited):
                assert self._dist[u][v] != 0
                child = pybnb.Node()
                child.state = (self._path + [v],)
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
    problem = TSP_ByVertex(dist)

    solver = pybnb.Solver()
    results = run_solve_loop(dist, problem, solver)
    stats = solver.collect_worker_statistics()
    if solver.is_dispatcher:
        pybnb.solver.summarize_worker_statistics(stats)
        # save results to a file
        # (mainly used for testing this example)
        if args.results_filename is not None:
            results.write(args.results_filename)
