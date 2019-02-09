#
# This example defines a script that solves the Traveling
# Salesperson Problem using a naive branching and bounding
# strategy. It highlights a number of advanced features,
# including:
#  (1) the use of the pybnb.futures.NestedSolver
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

def parse_dense_adjacency(filename):
    """Extracts a dense adjacency matrix from a file with
    the given name. Assumes columns are separated by
    whitespace and rows are separated by newlines. For
    consistency, entries that are zero on the off-diagonal
    will be converted to inf."""
    import math
    adj = []
    with open(filename) as f:
        line = f.readline().strip()
        while line:
            adj.append([float(x) for x in line.split()])
            line = f.readline().strip()
    N = len(adj)
    for i,row in enumerate(adj):
        assert len(row) == N
        for j,c in enumerate(row):
            assert c != -pybnb.inf
            assert not math.isnan(c)
            if i == j:
                assert c == 0
            elif c == 0:
                row[j] = pybnb.inf
    return adj

def run_2opt(adj, cost, tour):
    """Runs the 2-opt local search heuristic for TSP."""
    N = len(tour)
    assert N == len(adj)+1
    assert tour[0] == tour[-1]
    while (1):
        start_over = False
        for i in range(N-1):
            for k in range(i+1, N):
                new_cost = cost
                if i > 0:
                    if k < N-1:
                        new_cost -= adj[tour[i-1]][tour[i]]
                        new_cost += adj[tour[i-1]][tour[k]]
                        new_cost -= adj[tour[k]][tour[k+1]]
                        new_cost += adj[tour[i]][tour[k+1]]
                    else:
                        new_cost -= adj[tour[i-1]][tour[i]]
                        new_cost += adj[tour[i-1]][tour[k]]
                        new_cost -= adj[tour[0]][tour[k]]
                        new_cost += adj[tour[0]][tour[i]]
                else:
                    if k < N-1:
                        new_cost -= adj[tour[k]][tour[k+1]]
                        new_cost += adj[tour[i]][tour[k+1]]
                        new_cost -= adj[tour[i]][tour[-1]]
                        new_cost += adj[tour[k]][tour[-1]]
                    else:
                        continue
                if (new_cost < cost):
                    cost = new_cost
                    tour = (tour[:i] + \
                            list(reversed(tour[i:k+1])) +
                            tour[k+1:])
                    start_over = True
                    break
            if start_over:
                break
        if start_over:
            continue
        break
    return cost, tour

def run_3opt(adj, cost, tour):
    """Runs the 3-opt local search heuristic for TSP."""
    N = len(tour)
    assert N == len(adj)+1
    assert tour[0] == tour[-1]
    while (1):
        start_over = False
        for i in range(N-2):
            for k in range(i+2,N):
                for j in range(i,k):
                    new_cost = cost
                    if i > 0:
                        if k < N-1:
                            new_cost -= adj[tour[i-1]][tour[i]]
                            new_cost -= adj[tour[j]][tour[j+1]]
                            new_cost -= adj[tour[k]][tour[k+1]]
                            new_cost += adj[tour[i]][tour[k]]
                            new_cost += adj[tour[i-1]][tour[j+1]]
                            new_cost += adj[tour[j]][tour[k+1]]
                        else:
                            new_cost -= adj[tour[i-1]][tour[i]]
                            new_cost -= adj[tour[j]][tour[j+1]]
                            new_cost -= adj[tour[k]][tour[0]]
                            new_cost += adj[tour[i]][tour[k]]
                            new_cost += adj[tour[i-1]][tour[j+1]]
                            new_cost += adj[tour[j]][tour[0]]
                    else:
                        if k < N-1:
                            new_cost -= adj[tour[0]][tour[-1]]
                            new_cost -= adj[tour[j]][tour[j+1]]
                            new_cost -= adj[tour[k]][tour[k+1]]
                            new_cost += adj[tour[0]][tour[k]]
                            new_cost += adj[tour[j+1]][tour[-1]]
                            new_cost += adj[tour[j]][tour[k+1]]
                        else:
                            continue
                    if (new_cost < cost):
                        cost = new_cost
                        tour = (tour[0:i] + \
                                list(reversed(
                                    list(reversed(tour[i:j+1])) +
                                    list(reversed(tour[j+1:k+1])))) +
                                tour[k+1:])
                        start_over = True
                        break
            if start_over:
                break
        if start_over:
            continue
        break
    return cost, tour

class TSP_Naive(pybnb.Problem):

    def __init__(self, adj):
        self._adj = adj
        self._N = len(adj)
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
            return_cost = self._adj[self._path[-1]][self._path[0]]
            assert return_cost != 0
            if return_cost != pybnb.inf:
                cost = 0.0
                for i in range(self._N-1):
                    assert self._path[i] != -1
                    assert self._path[i+1] != -1
                    cost += self._adj[self._path[i]][self._path[i+1]]
                cost += self._adj[self._path[-1]][self._path[0]]
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
            bound += self._adj[self._path[i]][self._path[i+1]]
        # for the last item
        last = self._path[-1]
        tmp = [self._adj[last][v] for v in remaining
               if ((self._adj[last][v] != pybnb.inf) and \
                   (v != last))]
        if len(tmp) == 0:
            return self.infeasible_objective()
        bound += min(tmp)
        # for the undetermined nodes
        p = [self._path[0]] + remaining
        for r in remaining:
            tmp = [self._adj[r][v] for v in p
                   if ((self._adj[r][v] != pybnb.inf) and \
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
            # adj[u][v] == inf means no edge
            if (self._adj[u][v] != pybnb.inf) and \
               (v not in visited):
                assert self._adj[u][v] != 0
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
            tour_ = path + [path[0]]
            tour = {'cost': results.best_node.objective,
                    'tour': tour_}
        results.tour = tour

if __name__ == "__main__":
    import pybnb.futures
    import argparse

    parser = argparse.ArgumentParser(
        description=("Run parallel branch and bound "
                     "to solve an instance of TSP."))
    parser.add_argument("data_filename", type=str,
                        help=("The name of a file that stores a "
                              "dense adjacency matrix."))
    parser.add_argument("--results-filename", type=str, default=None,
                        help=("When set, saves the solver results "
                              "into a YAML-formated file with the "
                              "given name."))
    args = parser.parse_args()

    adj = parse_dense_adjacency(args.data_filename)
    problem = TSP_Naive(adj)
    solver = pybnb.Solver()

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
            queue_strategy='bound',
            initialize_queue=queue,
            best_node=best_node,
            objective_stop=objective_stop)
        if results.solution_status == "feasible":
           assert results.best_node is not None
           assert results.tour is not None
           cost, tour = run_3opt(
               adj,
               *run_2opt(adj,
                         results.tour['cost'],
                         results.tour['tour']))
           best_node = pybnb.Node()
           best_node.objective = cost
           best_node.state = tour[:-1]
           objective_stop = cost-1
           queue = solver.save_dispatcher_queue()
        else:
            if solver.is_dispatcher:
                print("Terminating the solve loop.")
                print("Final solution status: "
                      +str(results.solution_status))
            break

    if solver.is_dispatcher:
        # save results to a file
        # (mainly used for testing this example)
        if args.results_filename is not None:
            results.write(args.results_filename)
