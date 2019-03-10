import math

import pybnb

def parse_dense_distance_matrix(filename):
    """Extracts a dense distance matrix from a file with
    the given name. Assumes columns are separated by
    whitespace and rows are separated by newlines. For
    consistency, entries that are zero on the off-diagonal
    will be converted to inf."""
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

def run_solve_loop(dist, problem, solver):
    """Solves the TSP using a combination of
    branch-and-bound and local heuristics."""

    # The solve loop below does the following:
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
        results = solver.solve(
            pybnb.futures.NestedSolver(problem,
                                       queue_strategy='depth',
                                       track_bound=False,
                                       time_limit=1),
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
           best_node.state = (route[:-1],)
           objective_stop = cost - 1
           queue = solver.save_dispatcher_queue()
        else:
            if solver.is_dispatcher:
                print("Terminating the solve loop.")
                print("Final solution status: "
                      +str(results.solution_status))
                print("")
            break

    return results
