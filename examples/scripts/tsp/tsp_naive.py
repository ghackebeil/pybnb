#
# This example defines a script that solves the Traveling
# Salesperson Problem using a naive branching and bounding
# strategy. It highlights a number of advanced pybnb
# features, including:
#  (1) the use of pybnb.futures.NestedSolver
#  (2) re-continuing a solve after early termination
#
# This example can be executed in serial as
#
# $ python tsp_naive.py <data_file>
#
# or in parallel as
#
# $ mpiexec -n <n> python tsp_naive.py <data_file>
#
# The following data files are available:
# (source: https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html)
#  - p01_d.txt: 15 cities, minimal tour length 291
#  - p01_d_inf.txt: same as above, but with random paths
#                   removed to make the problem infeasible
#  - fri26_d.txt: 26 cities, minimal tour length 937
#
import pybnb

def parse_dense_distance_matrix(filename):
    """Extracts a dense distance matrix from a file with
    the given name. Assumes columns are separated by
    whitespace and rows are separated by newlines. For
    consistency, entries that are zero on the off-diagonal
    will be converted to inf."""
    import math
    dist = []
    with open(filename) as f:
        line = f.readline().strip()
        while line:
            dist.append([float(x) for x in line.split()])
            line = f.readline().strip()
    N = len(dist)
    for i,row in enumerate(dist):
        assert len(row) == N
        for j,c in enumerate(row):
            assert c != -pybnb.inf
            assert not math.isnan(c)
            if i == j:
                assert c == 0
            elif c == 0:
                row[j] = pybnb.inf
    return dist

def compute_route_cost(dist, route):
    """Compute the cost of a route."""
    N = len(route)
    assert N == len(dist)+1
    assert route[0] == route[-1]
    cost = 0
    for i in range(1,len(route)):
        u = route[i-1]
        v = route[i]
        c = dist[u][v]
        assert c != 0
        cost += c
    return cost

def run_2opt(dist, cost, route):
    """Runs the 2-opt local search heuristic for TSP. Does
    not assume the distance matrix is symmetric. This
    function makes a copy of the route argument before
    running the heuristic, so the input list is not
    modified."""
    N = len(route)
    assert N == len(dist)+1
    assert route[0] == route[-1]
    route = list(route)
    while (1):
        start_over = False
        for i in range(1,N-1):
            for j in range(i+1, N):
                if j-i == 1:
                    continue
                route[i:j] = route[j-1:i-1:-1]
                new_cost = compute_route_cost(dist, route)
                if new_cost < cost:
                    cost = new_cost
                    start_over = True
                    break
                else:
                    # reset the route
                    route[i:j] = route[j-1:i-1:-1]
            if start_over:
                break
        if start_over:
            continue
        break
    return cost, route

class TSP_Naive(pybnb.Problem):

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
        node.state = self._path

    def load_state(self, node):
        self._path = node.state
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
                child.state = self._path + [v]
                yield child

    def notify_solve_finished(self,
                              comm,
                              worker_comm,
                              results):
        tour = None
        if (results.best_node is not None) and \
           (results.best_node.state is not None):
            path = results.best_node.state
            route = path + [path[0]]
            tour = {'cost': results.best_node.objective,
                    'route': route}
        results.tour = tour

if __name__ == "__main__":
    import argparse

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
    problem = TSP_Naive(dist)
    solver = pybnb.Solver()

    # The following solve loop does the following:
    #  (1) Solve the tsp problem using a nested
    #      branch-and-bound strategy until any improvement
    #      to the previous best cost is made (objective_stop)
    #  (2) If the solution status from (1) is feasible, run
    #      the 2-opt heuristic to attempt to improve the
    #      solution. For any other solution status (e.g.,
    #      optimal, infeasible), exit the solve loop.
    #  (3) Go to step (1), initializing the solve with the
    #      remaining queue items from the previous solve
    #      (initialize_queue), a potentially new best node
    #      created with the solution returned from the 2-opt
    #      heuristic (best_node), and a new objective_stop
    #      value of one less than the current best cost (so
    #      we can go to step (2) if a new solution is
    #      found).
    objective_stop = pybnb.inf
    queue = None
    best_node = None
    while (1):
        # solve exactly (disable check for relative
        # gap and use absolute gap of zero)
        results = solver.solve(
            pybnb.futures.NestedSolver(problem,
                                       queue_strategy='depth',
                                       time_limit=1),
            absolute_gap=0,
            relative_gap=None,
            queue_strategy='depth',
            initialize_queue=queue,
            best_node=best_node,
            objective_stop=objective_stop)
        if (results.solution_status == "feasible") and \
           (results.termination_condition != "interrupted"):
           assert results.best_node is not None
           assert results.tour is not None
           cost, route = run_2opt(dist,
                                  results.tour['cost'],
                                  results.tour['route'])
           if cost < results.tour['cost']:
               if solver.is_dispatcher:
                   print("Local heuristic improved best tour:")
                   print(" -  cost: "+str(cost))
                   print(" - route: "+str(route))
           best_node = pybnb.Node()
           best_node.objective = cost
           best_node.state = route[:-1]
           objective_stop = cost - 1
           queue = solver.save_dispatcher_queue()
        else:
            if solver.is_dispatcher:
                print("Terminating the solve loop.")
                print("Final solution status: "
                      +str(results.solution_status))
            break

    stats = solver.collect_worker_statistics()
    if solver.is_dispatcher:
        pybnb.solver.summarize_worker_statistics(stats)
        # save results to a file
        # (mainly used for testing this example)
        if args.results_filename is not None:
            results.write(args.results_filename)
