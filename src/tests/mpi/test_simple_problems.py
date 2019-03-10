import os
import sys

from runtests.mpi import MPITest

from pybnb import (QueueStrategy,
                   Node,
                   inf,
                   SolverResults,
                   Solver)
from pybnb.misc import get_simple_logger
from pybnb.futures import NestedSolver

from .common import mpi_available

thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thisdir)
try:
    from problems import (infeasible_max,
                          infeasible_min,
                          root_infeasible_max,
                          root_infeasible_min,
                          unbounded_max,
                          unbounded_min,
                          zero_objective_max,
                          zero_objective_min,
                          delayed_unbounded_max,
                          delayed_unbounded_min)

finally:
    sys.path.remove(thisdir)

_ignore_value_ = object()

_queue_strategies = sorted(QueueStrategy)
_queue_strategies.append((QueueStrategy.bound,
                          QueueStrategy.objective,
                          'breadth'))

import six

def _execute_single_test(problem,
                         baseline,
                         solver=None,
                         comm=None,
                         **kwds):
    if solver is None:
        solver = Solver(comm=comm)
    else:
        assert comm is None
    if solver.comm is not None:
        if solver.comm.rank == 0:
            pass
        elif solver.comm.rank == 1:
            pass
        elif solver.comm.rank == 3:
            pass
    orig = Node()
    problem.save_state(orig)
    results = solver.solve(problem, **kwds)
    assert isinstance(results.solution_status,
                      six.string_types)
    assert isinstance(results.termination_condition,
                      six.string_types)
    current = Node()
    problem.save_state(current)
    assert current.state == orig.state
    assert len(vars(results)) > 0
    assert len(vars(results)) == len(vars(baseline))
    for name in sorted(sorted(list(vars(results).keys()))):
        if getattr(baseline, name) is _ignore_value_:
            continue
        if (name == 'nodes') and \
           (solver.comm is not None) and \
           (solver.comm.size > 2):
            pass
        else:
            assert getattr(results, name) == getattr(baseline, name), \
                ("value for '"+str(name)+"' ("+
                 str(getattr(results, name))+") does "
                 "not match baseline ("+
                 str(getattr(baseline, name))+")")
    if solver.is_dispatcher:
        q = solver.save_dispatcher_queue()
        assert len(q.nodes) == solver._disp.queue.size()
        assert q.sense == solver._disp.converger.sense
        assert q.worst_terminal_bound == solver._disp.worst_terminal_bound
        assert q.bound() == results.bound
    return results

def _execute_tests(comm, problem, baseline, **kwds):
    assert 'log_interval_second' not in kwds
    assert 'log' not in kwds
    _execute_single_test(problem, baseline, comm=comm, **kwds)
    kwds['log_interval_seconds'] = 0.0
    _execute_single_test(problem, baseline, comm=comm, **kwds)
    kwds['log_interval_seconds'] = 100.0
    _execute_single_test(problem, baseline, comm=comm, **kwds)
    kwds['log_interval_seconds'] = 0.0
    kwds['log'] = None
    _execute_single_test(problem, baseline, comm=comm, **kwds)
    kwds['log_interval_seconds'] = 100.0
    _execute_single_test(problem, baseline, comm=comm, **kwds)
    solver = Solver(comm=comm)
    _execute_single_test(problem, baseline, solver=solver, **kwds)
    _execute_single_test(problem, baseline, solver=solver, **kwds)
    kwds['log'] = get_simple_logger(level="WARNING")
    _execute_single_test(problem, baseline, solver=solver, **kwds)

def _test_infeasible_max(comm):
    solver = None
    if comm is not None:
        solver = Solver(comm=comm)

    baseline = SolverResults()
    baseline.solution_status = "infeasible"
    baseline.termination_condition = "optimality"
    baseline.objective = -inf
    baseline.bound = -inf
    baseline.nodes = 255
    baseline.best_node = None
    baseline.wall_time = _ignore_value_
    problem = infeasible_max.InfeasibleMax()
    if comm is None:
        _execute_tests(comm,
                       problem,
                       baseline)
    else:
        _execute_single_test(problem,
                             baseline,
                             solver=solver)


    baseline = SolverResults()
    baseline.solution_status = "unknown"
    baseline.termination_condition = "queue_empty"
    baseline.objective = -inf
    baseline.bound = -16
    baseline.nodes = 31
    baseline.best_node = None
    baseline.wall_time = _ignore_value_
    problem = infeasible_max.InfeasibleMax(branching_abstol=0.1)
    if comm is None:
        _execute_tests(comm,
                       problem,
                       baseline)
    else:
        _execute_single_test(problem,
                             baseline,
                             solver=solver)

    baseline = SolverResults()
    baseline.solution_status = "unknown"
    baseline.termination_condition = "objective_limit"
    baseline.objective = -inf
    baseline.bound = -16
    baseline.nodes = 31
    baseline.best_node = None
    baseline.wall_time = _ignore_value_
    problem = infeasible_max.InfeasibleMax(branching_abstol=0.1)
    if comm is None:
        _execute_tests(comm,
                       problem,
                       baseline,
                       bound_stop=-15)
    else:
        _execute_single_test(problem,
                             baseline,
                             solver=solver,
                             bound_stop=-15)

    baseline = SolverResults()
    baseline.solution_status = "feasible"
    baseline.termination_condition = "objective_limit"
    baseline.objective = -20
    baseline.bound = -16
    baseline.absolute_gap = 4
    baseline.relative_gap = 0.2
    baseline.nodes = 31
    baseline.best_node = _ignore_value_
    baseline.wall_time = _ignore_value_
    problem = infeasible_max.InfeasibleMax(branching_abstol=0.1,
                                           fixed_objective=-20)
    if comm is None:
        _execute_tests(comm,
                       problem,
                       baseline,
                       bound_stop=-15)
    else:
        _execute_single_test(problem,
                             baseline,
                             solver=solver,
                             bound_stop=-15)

    baseline = SolverResults()
    baseline.solution_status = "unknown"
    baseline.termination_condition = "node_limit"
    baseline.objective = -inf
    baseline.bound = -16
    baseline.nodes = 31
    baseline.best_node = None
    baseline.wall_time = _ignore_value_
    problem = infeasible_max.InfeasibleMax()
    if comm is None:
        _execute_tests(comm,
                       problem,
                       baseline,
                       node_limit=31)
    elif comm.size <= 2:
        # skip for larger comm sizes
        # as the node_limit can lead to
        # a number of different outcomes
        _execute_single_test(problem,
                             baseline,
                             solver=solver,
                             node_limit=31)

    baseline = SolverResults()
    baseline.solution_status = "feasible"
    baseline.termination_condition = "node_limit"
    baseline.objective = -17
    baseline.bound = -16
    baseline.absolute_gap = 1.0
    baseline.relative_gap = 1.0/17
    baseline.nodes = 31
    baseline.best_node = _ignore_value_
    baseline.wall_time = _ignore_value_
    problem = infeasible_max.InfeasibleMax(fixed_objective=-17)
    if comm is None:
        _execute_tests(comm,
                       problem,
                       baseline,
                       node_limit=31)
    elif comm.size <= 2:
        # skip for larger comm sizes
        # as the node_limit can lead to
        # a number of different outcomes
        _execute_single_test(problem,
                             baseline,
                             solver=solver,
                             node_limit=31)

    baseline = SolverResults()
    baseline.solution_status = "unknown"
    baseline.termination_condition = "time_limit"
    baseline.objective = -inf
    baseline.bound = inf
    baseline.nodes = 0
    baseline.best_node = None
    baseline.wall_time = _ignore_value_
    problem = infeasible_max.InfeasibleMax()
    if comm is None:
        _execute_tests(comm,
                       problem,
                       baseline,
                       time_limit=0)
    else:
        _execute_single_test(problem,
                             baseline,
                             solver=solver,
                             time_limit=0)

    if solver is None:
        solver = Solver(comm=comm)
    baseline = SolverResults()
    baseline.solution_status = "infeasible"
    baseline.termination_condition = "optimality"
    baseline.objective = -inf
    baseline.bound = -inf
    baseline.nodes = 255
    baseline.best_node = None
    baseline.wall_time = _ignore_value_
    problem = infeasible_max.InfeasibleMax()
    for queue_strategy in _queue_strategies:
        if queue_strategy == "custom":
            continue
        _execute_single_test(
            problem,
            baseline,
            solver=solver,
            queue_strategy=queue_strategy)

    baseline = SolverResults()
    baseline.solution_status = "invalid"
    baseline.termination_condition = "optimality"
    baseline.objective = -100000000
    baseline.bound = -inf
    baseline.nodes = _ignore_value_
    baseline.best_node = None
    baseline.wall_time = _ignore_value_
    problem = infeasible_max.InfeasibleMax()
    for queue_strategy in _queue_strategies:
        if queue_strategy == "custom":
            continue
        _execute_single_test(
            problem,
            baseline,
            solver=solver,
            queue_strategy=queue_strategy,
            best_objective=-100000000)

    baseline1 = SolverResults()
    baseline1.solution_status = "infeasible"
    baseline1.termination_condition = "optimality"
    baseline1.objective = -inf
    baseline1.bound = -inf
    baseline1.nodes = _ignore_value_
    baseline1.best_node = None
    baseline1.wall_time = _ignore_value_
    baseline2 = SolverResults()
    baseline2.solution_status = "infeasible"
    baseline2.termination_condition = "optimality"
    baseline2.objective = -inf
    baseline2.bound = -inf
    baseline2.nodes = _ignore_value_
    baseline2.best_node = None
    baseline2.wall_time = _ignore_value_
    problem = infeasible_max.InfeasibleMax()
    for queue_strategy in _queue_strategies:
        if queue_strategy == "custom":
            continue
        results = _execute_single_test(
            problem,
            baseline1,
            solver=solver,
            queue_strategy=queue_strategy)
        queue = solver.save_dispatcher_queue()
        _execute_single_test(
            problem,
            baseline2,
            solver=solver,
            queue_strategy=queue_strategy,
            initialize_queue=queue,
            best_objective=results.objective,
            best_node=results.best_node)

def _test_infeasible_min(comm):
    solver = None
    if comm is not None:
        solver = Solver(comm=comm)
    baseline = SolverResults()
    baseline.solution_status = "infeasible"
    baseline.termination_condition = "optimality"
    baseline.objective = inf
    baseline.bound = inf
    baseline.nodes = 255
    baseline.best_node = None
    baseline.wall_time = _ignore_value_
    problem = infeasible_min.InfeasibleMin()
    if comm is None:
        _execute_tests(comm,
                       problem,
                       baseline)
    else:
        _execute_single_test(problem,
                             baseline,
                             solver=solver)

    baseline = SolverResults()
    baseline.solution_status = "unknown"
    baseline.termination_condition = "queue_empty"
    baseline.objective = inf
    baseline.bound = 16
    baseline.nodes = 31
    baseline.best_node = None
    baseline.wall_time = _ignore_value_
    problem = infeasible_min.InfeasibleMin(branching_abstol=0.1)
    if comm is None:
        _execute_tests(comm,
                       problem,
                       baseline)
    else:
        _execute_single_test(problem,
                             baseline,
                             solver=solver)

    baseline = SolverResults()
    baseline.solution_status = "unknown"
    baseline.termination_condition = "objective_limit"
    baseline.objective = inf
    baseline.bound = 16
    baseline.nodes = 31
    baseline.best_node = None
    baseline.wall_time = _ignore_value_
    problem = infeasible_min.InfeasibleMin(branching_abstol=0.1)
    if comm is None:
        _execute_tests(comm,
                       problem,
                       baseline,
                       bound_stop=15)
    else:
        _execute_single_test(problem,
                             baseline,
                             solver=solver,
                             bound_stop=15)

    baseline = SolverResults()
    baseline.solution_status = "feasible"
    baseline.termination_condition = "objective_limit"
    baseline.objective = 20
    baseline.bound = 16
    baseline.absolute_gap = 4
    baseline.relative_gap = 0.2
    baseline.nodes = 31
    baseline.best_node = _ignore_value_
    baseline.wall_time = _ignore_value_
    problem = infeasible_min.InfeasibleMin(branching_abstol=0.1,
                                           fixed_objective=20)
    if comm is None:
        _execute_tests(comm,
                       problem,
                       baseline,
                       bound_stop=15)
    else:
        _execute_single_test(problem,
                             baseline,
                             solver=solver,
                             bound_stop=15)

    baseline = SolverResults()
    baseline.solution_status = "unknown"
    baseline.termination_condition = "node_limit"
    baseline.objective = inf
    baseline.bound = 16
    baseline.nodes = 31
    baseline.best_node = None
    baseline.wall_time = _ignore_value_
    problem = infeasible_min.InfeasibleMin()
    if comm is None:
        _execute_tests(comm,
                       problem,
                       baseline,
                       node_limit=31)
    elif comm.size <= 2:
        # skip for larger comm sizes
        # as the node_limit can lead to
        # a number of different outcomes
        _execute_single_test(problem,
                             baseline,
                             solver=solver,
                             node_limit=31)

    baseline = SolverResults()
    baseline.solution_status = "feasible"
    baseline.termination_condition = "node_limit"
    baseline.objective = 17
    baseline.bound = 16
    baseline.absolute_gap = 1
    baseline.relative_gap = 1.0/17
    baseline.nodes = 31
    baseline.best_node = _ignore_value_
    baseline.wall_time = _ignore_value_
    problem = infeasible_min.InfeasibleMin(fixed_objective=17)
    if comm is None:
        _execute_tests(comm,
                       problem,
                       baseline,
                       node_limit=31)
    elif comm.size <= 2:
        # skip for larger comm sizes
        # as the node_limit can lead to
        # a number of different outcomes
        _execute_single_test(problem,
                             baseline,
                             solver=solver,
                             node_limit=31)

    baseline = SolverResults()
    baseline.solution_status = "unknown"
    baseline.termination_condition = "time_limit"
    baseline.objective = inf
    baseline.bound = -inf
    baseline.nodes = 0
    baseline.best_node = None
    baseline.wall_time = _ignore_value_
    problem = infeasible_min.InfeasibleMin()
    if comm is None:
        _execute_tests(comm,
                       problem,
                       baseline,
                       time_limit=0)
    else:
        _execute_single_test(problem,
                             baseline,
                             solver=solver,
                             time_limit=0)

    if solver is None:
        solver = Solver(comm=comm)
    baseline = SolverResults()
    baseline.solution_status = "infeasible"
    baseline.termination_condition = "optimality"
    baseline.objective = inf
    baseline.bound = inf
    baseline.nodes = 255
    baseline.best_node = None
    baseline.wall_time = _ignore_value_
    problem = infeasible_min.InfeasibleMin()
    for queue_strategy in _queue_strategies:
        if queue_strategy == "custom":
            continue
        _execute_single_test(
            problem,
            baseline,
            solver=solver,
            queue_strategy=queue_strategy)

    baseline = SolverResults()
    baseline.solution_status = "invalid"
    baseline.termination_condition = "optimality"
    baseline.objective = 100000000
    baseline.bound = inf
    baseline.nodes = _ignore_value_
    baseline.best_node = None
    baseline.wall_time = _ignore_value_
    problem = infeasible_min.InfeasibleMin()
    for queue_strategy in _queue_strategies:
        if queue_strategy == "custom":
            continue
        _execute_single_test(
            problem,
            baseline,
            solver=solver,
            queue_strategy=queue_strategy,
            best_objective=100000000)

    baseline1 = SolverResults()
    baseline1.solution_status = "infeasible"
    baseline1.termination_condition = "optimality"
    baseline1.objective = inf
    baseline1.bound = inf
    baseline1.nodes = _ignore_value_
    baseline1.best_node = None
    baseline1.wall_time = _ignore_value_
    baseline2 = SolverResults()
    baseline2.solution_status = "infeasible"
    baseline2.termination_condition = "optimality"
    baseline2.objective = inf
    baseline2.bound = inf
    baseline2.nodes = _ignore_value_
    baseline2.best_node = None
    baseline2.wall_time = _ignore_value_
    problem = infeasible_min.InfeasibleMin()
    for queue_strategy in _queue_strategies:
        if queue_strategy == "custom":
            continue
        results = _execute_single_test(
            problem,
            baseline1,
            solver=solver,
            queue_strategy=queue_strategy)
        queue = solver.save_dispatcher_queue()
        _execute_single_test(
            problem,
            baseline2,
            solver=solver,
            queue_strategy=queue_strategy,
            initialize_queue=queue,
            best_objective=results.objective,
            best_node=results.best_node)

def _test_root_infeasible_max(comm):
    solver = Solver(comm=comm)
    baseline1 = SolverResults()
    baseline1.solution_status = "infeasible"
    baseline1.termination_condition = "optimality"
    baseline1.objective = -inf
    baseline1.bound = -inf
    baseline1.nodes = 1
    baseline1.best_node = None
    baseline1.wall_time = _ignore_value_
    baseline2 = SolverResults()
    baseline2.solution_status = "infeasible"
    baseline2.termination_condition = "optimality"
    baseline2.objective = -inf
    baseline2.bound = -inf
    baseline2.nodes = 0
    baseline2.best_node = None
    baseline2.wall_time = _ignore_value_
    problem = root_infeasible_max.RootInfeasibleMax()
    for queue_strategy in _queue_strategies:
        if queue_strategy == "custom":
            continue
        results = _execute_single_test(
            problem,
            baseline1,
            solver=solver,
            queue_strategy=queue_strategy)
        queue = solver.save_dispatcher_queue()
        _execute_single_test(
            problem,
            baseline2,
            solver=solver,
            queue_strategy=queue_strategy,
            initialize_queue=queue,
            best_objective=results.objective,
            best_node=results.best_node)

def _test_root_infeasible_min(comm):
    solver = Solver(comm=comm)
    baseline1 = SolverResults()
    baseline1.solution_status = "infeasible"
    baseline1.termination_condition = "optimality"
    baseline1.objective = inf
    baseline1.bound = inf
    baseline1.nodes = 1
    baseline1.best_node = None
    baseline1.wall_time = _ignore_value_
    baseline2 = SolverResults()
    baseline2.solution_status = "infeasible"
    baseline2.termination_condition = "optimality"
    baseline2.objective = inf
    baseline2.bound = inf
    baseline2.nodes = 0
    baseline2.best_node = None
    baseline2.wall_time = _ignore_value_
    problem = root_infeasible_min.RootInfeasibleMin()
    for queue_strategy in _queue_strategies:
        if queue_strategy == "custom":
            continue
        results = _execute_single_test(
            problem,
            baseline1,
            solver=solver,
            queue_strategy=queue_strategy)
        queue = solver.save_dispatcher_queue()
        _execute_single_test(
            problem,
            baseline2,
            solver=solver,
            queue_strategy=queue_strategy,
            initialize_queue=queue,
            best_objective=results.objective,
            best_node=results.best_node)

def _test_unbounded_max(comm):
    solver = Solver(comm=comm)

    baseline = SolverResults()
    baseline.solution_status = "unbounded"
    baseline.termination_condition = "optimality"
    baseline.objective = inf
    baseline.bound = inf
    baseline.nodes = 1
    baseline.best_node = None
    baseline.wall_time = _ignore_value_
    problem = unbounded_max.UnboundedMax()
    _execute_single_test(problem,
                         baseline,
                         solver=solver)
    for queue_strategy in _queue_strategies:
        if queue_strategy == "custom":
            continue
        _execute_single_test(
            problem,
            baseline,
            solver=solver,
            queue_strategy=queue_strategy)

    baseline1 = SolverResults()
    baseline1.solution_status = "unbounded"
    baseline1.termination_condition = "optimality"
    baseline1.objective = inf
    baseline1.bound = inf
    baseline1.nodes = 1
    baseline1.best_node = None
    baseline1.wall_time = _ignore_value_
    baseline2 = SolverResults()
    baseline2.solution_status = "unbounded"
    baseline2.termination_condition = "optimality"
    baseline2.objective = inf
    baseline2.bound = inf
    baseline2.nodes = 0
    baseline2.best_node = None
    baseline2.wall_time = _ignore_value_
    problem = unbounded_max.UnboundedMax()
    for queue_strategy in _queue_strategies:
        if queue_strategy == "custom":
            continue
        results = _execute_single_test(
            problem,
            baseline1,
            solver=solver,
            queue_strategy=queue_strategy)
        queue = solver.save_dispatcher_queue()
        _execute_single_test(
            problem,
            baseline2,
            solver=solver,
            queue_strategy=queue_strategy,
            initialize_queue=queue,
            best_objective=results.objective,
            best_node=results.best_node)

def _test_unbounded_min(comm):
    solver = Solver(comm=comm)

    baseline = SolverResults()
    baseline.solution_status = "unbounded"
    baseline.termination_condition = "optimality"
    baseline.objective = -inf
    baseline.bound = -inf
    baseline.nodes = 1
    baseline.best_node = None
    baseline.wall_time = _ignore_value_
    problem = unbounded_min.UnboundedMin()
    _execute_single_test(problem,
                         baseline,
                         solver=solver)
    for queue_strategy in _queue_strategies:
        if queue_strategy == "custom":
            continue
        _execute_single_test(
            problem,
            baseline,
            solver=solver,
            queue_strategy=queue_strategy)

    baseline1 = SolverResults()
    baseline1.solution_status = "unbounded"
    baseline1.termination_condition = "optimality"
    baseline1.objective = -inf
    baseline1.bound = -inf
    baseline1.nodes = 1
    baseline1.best_node = None
    baseline1.wall_time = _ignore_value_
    baseline2 = SolverResults()
    baseline2.solution_status = "unbounded"
    baseline2.termination_condition = "optimality"
    baseline2.objective = -inf
    baseline2.bound = -inf
    baseline2.nodes = 0
    baseline2.best_node = None
    baseline2.wall_time = _ignore_value_
    problem = unbounded_min.UnboundedMin()
    for queue_strategy in _queue_strategies:
        if queue_strategy == "custom":
            continue
        results = _execute_single_test(
            problem,
            baseline1,
            solver=solver,
            queue_strategy=queue_strategy)
        queue = solver.save_dispatcher_queue()
        _execute_single_test(
            problem,
            baseline2,
            solver=solver,
            queue_strategy=queue_strategy,
            initialize_queue=queue,
            best_objective=results.objective,
            best_node=results.best_node)

def _test_zero_objective_max(comm):
    solver = Solver(comm=comm)
    baseline = SolverResults()
    baseline.solution_status = "optimal"
    baseline.termination_condition = "optimality"
    baseline.objective = 0.0
    baseline.bound = 0.0078125
    baseline.absolute_gap = 0.0078125
    baseline.relative_gap = 0.0078125
    baseline.nodes = 255
    baseline.best_node = _ignore_value_
    baseline.wall_time = _ignore_value_
    problem = zero_objective_max.ZeroObjectiveMax()
    _execute_single_test(problem,
                         baseline,
                         solver=solver,
                         relative_gap=0.01,
                         absolute_gap=0.01)
    _execute_single_test(problem,
                         baseline,
                         solver=solver,
                         relative_gap=0.0,
                         absolute_gap=0.01)
    _execute_single_test(problem,
                         baseline,
                         solver=solver,
                         relative_gap=0.01,
                         absolute_gap=0.0)

    baseline = SolverResults()
    baseline.solution_status = "optimal"
    baseline.termination_condition = "optimality"
    baseline.objective = 0.0
    baseline.bound = 0.0009765625
    baseline.absolute_gap = 0.0009765625
    baseline.relative_gap = 0.0009765625
    baseline.nodes = 2047
    baseline.best_node = _ignore_value_
    baseline.wall_time = _ignore_value_
    problem = zero_objective_max.ZeroObjectiveMax()
    _execute_single_test(problem,
                         baseline,
                         solver=solver,
                         relative_gap=0.001,
                         absolute_gap=0.001)
    _execute_single_test(problem,
                         baseline,
                         solver=solver,
                         relative_gap=0.0,
                         absolute_gap=0.001)
    _execute_single_test(problem,
                         baseline,
                         solver=solver,
                         relative_gap=0.001,
                         absolute_gap=0.0)

    baseline = SolverResults()
    baseline.solution_status = "feasible"
    baseline.termination_condition = "queue_empty"
    baseline.objective = 0.0
    baseline.bound = 0.0009765625
    baseline.absolute_gap = 0.0009765625
    baseline.relative_gap = 0.0009765625
    baseline.nodes = 2047
    baseline.best_node = _ignore_value_
    baseline.wall_time = _ignore_value_
    problem = zero_objective_max.ZeroObjectiveMax()
    _execute_single_test(problem,
                         baseline,
                         solver=solver,
                         relative_gap=0.0001,
                         absolute_gap=0.0001)
    _execute_single_test(problem,
                         baseline,
                         solver=solver,
                         relative_gap=0.0,
                         absolute_gap=0.0001)
    _execute_single_test(problem,
                         baseline,
                         solver=solver,
                         relative_gap=0.0001,
                         absolute_gap=0.0)

    baseline = SolverResults()
    baseline.solution_status = "optimal"
    baseline.termination_condition = "optimality"
    baseline.objective = 0.0
    baseline.bound = 0.0078125
    baseline.absolute_gap = 0.0078125
    baseline.relative_gap = 0.0078125
    baseline.nodes = None
    baseline.best_node = _ignore_value_
    baseline.wall_time = _ignore_value_
    problem = zero_objective_max.ZeroObjectiveMax()
    for queue_strategy in _queue_strategies:
        if queue_strategy == "custom":
            continue
        elif queue_strategy == "random":
            baseline.nodes = _ignore_value_
        else:
            baseline.nodes = 255
        _execute_single_test(
            problem,
            baseline,
            solver=solver,
            relative_gap=0.01,
            absolute_gap=0.01,
            queue_strategy=queue_strategy)

def _test_zero_objective_min(comm):
    solver = Solver(comm=comm)
    baseline = SolverResults()
    baseline.solution_status = "optimal"
    baseline.termination_condition = "optimality"
    baseline.objective = 0.0
    baseline.bound = -0.0078125
    baseline.absolute_gap = 0.0078125
    baseline.relative_gap = 0.0078125
    baseline.nodes = 255
    baseline.best_node = _ignore_value_
    baseline.wall_time = _ignore_value_
    problem = zero_objective_min.ZeroObjectiveMin()
    _execute_single_test(problem,
                         baseline,
                         solver=solver,
                         relative_gap=0.01,
                         absolute_gap=0.01)
    _execute_single_test(problem,
                         baseline,
                         solver=solver,
                         relative_gap=0.0,
                         absolute_gap=0.01)
    _execute_single_test(problem,
                         baseline,
                         solver=solver,
                         relative_gap=0.01,
                         absolute_gap=0.0)

    baseline = SolverResults()
    baseline.solution_status = "optimal"
    baseline.termination_condition = "optimality"
    baseline.objective = 0.0
    baseline.bound = -0.0009765625
    baseline.absolute_gap = 0.0009765625
    baseline.relative_gap = 0.0009765625
    baseline.nodes = 2047
    baseline.best_node = _ignore_value_
    baseline.wall_time = _ignore_value_
    problem = zero_objective_min.ZeroObjectiveMin()
    _execute_single_test(problem,
                         baseline,
                         solver=solver,
                         relative_gap=0.001,
                         absolute_gap=0.001)
    _execute_single_test(problem,
                         baseline,
                         solver=solver,
                         relative_gap=0.0,
                         absolute_gap=0.001)
    _execute_single_test(problem,
                         baseline,
                         solver=solver,
                         relative_gap=0.001,
                         absolute_gap=0.0)

    baseline = SolverResults()
    baseline.solution_status = "feasible"
    baseline.termination_condition = "queue_empty"
    baseline.objective = 0.0
    baseline.bound = -0.0009765625
    baseline.absolute_gap = 0.0009765625
    baseline.relative_gap = 0.0009765625
    baseline.nodes = 2047
    baseline.best_node = _ignore_value_
    baseline.wall_time = _ignore_value_
    problem = zero_objective_min.ZeroObjectiveMin()
    _execute_single_test(problem,
                         baseline,
                         solver=solver,
                         relative_gap=0.0001,
                         absolute_gap=0.0001)
    _execute_single_test(problem,
                         baseline,
                         solver=solver,
                         relative_gap=0.0,
                         absolute_gap=0.0001)
    _execute_single_test(problem,
                         baseline,
                         solver=solver,
                         relative_gap=0.0001,
                         absolute_gap=0.0)

    baseline = SolverResults()
    baseline.solution_status = "optimal"
    baseline.termination_condition = "optimality"
    baseline.objective = 0.0
    baseline.bound = -0.0078125
    baseline.absolute_gap = 0.0078125
    baseline.relative_gap = 0.0078125
    baseline.nodes = None
    baseline.best_node = _ignore_value_
    baseline.wall_time = _ignore_value_
    problem = zero_objective_min.ZeroObjectiveMin()
    for queue_strategy in _queue_strategies:
        if queue_strategy == "custom":
            continue
        elif queue_strategy == "random":
            baseline.nodes = _ignore_value_
        else:
            baseline.nodes = 255
        _execute_single_test(
            problem,
            baseline,
            solver=solver,
            relative_gap=0.01,
            absolute_gap=0.01,
            queue_strategy=queue_strategy)

def _test_delayed_unbounded_max(comm):
    solver = Solver(comm=comm)
    baseline1 = SolverResults()
    baseline1.solution_status = "unbounded"
    baseline1.termination_condition = "optimality"
    baseline1.objective = inf
    baseline1.bound = inf
    baseline1.nodes = _ignore_value_
    baseline1.best_node = None
    baseline1.wall_time = _ignore_value_
    baseline2 = SolverResults()
    baseline2.solution_status = "unbounded"
    baseline2.termination_condition = "optimality"
    baseline2.objective = inf
    baseline2.bound = inf
    baseline2.nodes = 0
    baseline2.best_node = None
    baseline2.wall_time = _ignore_value_
    problem = delayed_unbounded_max.DelayedUnboundedMax()
    for queue_strategy in _queue_strategies:
        if queue_strategy == "custom":
            continue
        results = _execute_single_test(
            problem,
            baseline1,
            solver=solver,
            queue_strategy=queue_strategy)
        queue = solver.save_dispatcher_queue()
        _execute_single_test(
            problem,
            baseline2,
            solver=solver,
            queue_strategy=queue_strategy,
            initialize_queue=queue,
            best_objective=results.objective,
            best_node=results.best_node)
    # test the NestedSolver
    results = _execute_single_test(
        NestedSolver(problem,
                     time_limit=None,
                     node_limit=2),
        baseline1,
        solver=solver)
    queue = solver.save_dispatcher_queue()
    _execute_single_test(
        NestedSolver(problem,
                     time_limit=None,
                     node_limit=2),
        baseline2,
        solver=solver,
        initialize_queue=queue,
        best_objective=results.objective,
        best_node=results.best_node)

def _test_delayed_unbounded_min(comm):
    solver = Solver(comm=comm)
    baseline1 = SolverResults()
    baseline1.solution_status = "unbounded"
    baseline1.termination_condition = "optimality"
    baseline1.objective = -inf
    baseline1.bound = -inf
    baseline1.nodes = _ignore_value_
    baseline1.best_node = None
    baseline1.wall_time = _ignore_value_
    baseline2 = SolverResults()
    baseline2.solution_status = "unbounded"
    baseline2.termination_condition = "optimality"
    baseline2.objective = -inf
    baseline2.bound = -inf
    baseline2.nodes = 0
    baseline2.best_node = None
    baseline2.wall_time = _ignore_value_
    problem = delayed_unbounded_min.DelayedUnboundedMin()
    for queue_strategy in _queue_strategies:
        if queue_strategy == "custom":
            continue
        results = _execute_single_test(
            problem,
            baseline1,
            solver=solver,
            queue_strategy=queue_strategy)
        queue = solver.save_dispatcher_queue()
        _execute_single_test(
            problem,
            baseline2,
            solver=solver,
            queue_strategy=queue_strategy,
            initialize_queue=queue,
            best_objective=results.objective,
            best_node=results.best_node)
    # test the NestedSolver
    results = _execute_single_test(
        NestedSolver(problem,
                     time_limit=None,
                     node_limit=None),
        baseline1,
        solver=solver)
    queue = solver.save_dispatcher_queue()
    _execute_single_test(
        NestedSolver(problem,
                     time_limit=None,
                     node_limit=None),
        baseline2,
        solver=solver,
        initialize_queue=queue,
        best_objective=results.objective,
        best_node=results.best_node)

def test_infeasible_max_nocomm():
    _test_infeasible_max(None)

def test_infeasible_min_nocomm():
    _test_infeasible_min(None)

def test_root_infeasible_max_nocomm():
    _test_root_infeasible_max(None)

def test_root_infeasible_min_nocomm():
    _test_root_infeasible_min(None)

def test_unbounded_max_nocomm():
    _test_unbounded_max(None)

def test_unbounded_min_nocomm():
    _test_unbounded_min(None)

def test_zero_objective_max_nocomm():
    _test_zero_objective_max(None)

def test_zero_objective_min_nocomm():
    _test_zero_objective_min(None)

def test_delayed_unbounded_max_nocomm():
    _test_delayed_unbounded_max(None)

def test_delayed_unbounded_min_nocomm():
    _test_delayed_unbounded_min(None)

if mpi_available:

    @MPITest(commsize=[1, 2, 4])
    def test_infeasible_max(comm):
        _test_infeasible_max(comm)

    @MPITest(commsize=[1, 2, 4])
    def test_infeasible_min(comm):
        _test_infeasible_min(comm)

    @MPITest(commsize=[1, 2, 4])
    def test_root_infeasible_max(comm):
        _test_root_infeasible_max(comm)

    @MPITest(commsize=[1, 2, 4])
    def test_root_infeasible_min(comm):
        _test_root_infeasible_min(comm)

    @MPITest(commsize=[1, 2, 4])
    def test_unbounded_max(comm):
        _test_unbounded_max(comm)

    @MPITest(commsize=[1, 2, 4])
    def test_unbounded_min(comm):
        _test_unbounded_min(comm)

    @MPITest(commsize=[1, 2, 4])
    def test_zero_objective_max(comm):
        _test_zero_objective_max(comm)

    @MPITest(commsize=[1, 2, 4])
    def test_zero_objective_min(comm):
        _test_zero_objective_min(comm)

    @MPITest(commsize=[1, 2, 4])
    def test_delayed_unbounded_max(comm):
        _test_delayed_unbounded_max(comm)

    @MPITest(commsize=[1, 2, 4])
    def test_delayed_unbounded_min(comm):
        _test_delayed_unbounded_min(comm)
