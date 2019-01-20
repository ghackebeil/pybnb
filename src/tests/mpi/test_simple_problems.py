import os
import sys

import pytest
from runtests.mpi import MPITest

import pybnb
from pybnb.common import inf
from pybnb.solver import (SolverResults,
                          Solver)
from pybnb.misc import get_simple_logger

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
    orig = pybnb.node.Node()
    problem.save_state(orig)
    results = solver.solve(problem, **kwds)

    current = pybnb.node.Node()
    problem.save_state(current)
    assert len(current.state) == len(orig.state)
    for i in range(len(current.state)):
        assert current.state[i] == orig.state[i]
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
        assert len(q) == 2
        assert q.next_tree_id >= 0
        assert len(q.nodes) == solver._disp.queue.size()

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
    baseline.termination_condition = "no_nodes"
    baseline.objective = -inf
    baseline.bound = -16
    baseline.nodes = 31
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
    baseline.wall_time = _ignore_value_
    problem = infeasible_max.InfeasibleMax()
    for queue_strategy in sorted(pybnb.QueueStrategy):
        if queue_strategy == "custom":
            continue
        _execute_single_test(
            problem,
            baseline,
            solver=solver,
            queue_strategy=queue_strategy)

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
    baseline.termination_condition = "no_nodes"
    baseline.objective = inf
    baseline.bound = 16
    baseline.nodes = 31
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
    baseline.wall_time = _ignore_value_
    problem = infeasible_min.InfeasibleMin()
    for queue_strategy in sorted(pybnb.QueueStrategy):
        if queue_strategy == "custom":
            continue
        _execute_single_test(
            problem,
            baseline,
            solver=solver,
            queue_strategy=queue_strategy)

def _test_root_infeasible_max(comm):
    solver = Solver(comm=comm)
    baseline = SolverResults()
    baseline.solution_status = "infeasible"
    baseline.termination_condition = "optimality"
    baseline.objective = -inf
    baseline.bound = -inf
    baseline.nodes = 1
    baseline.wall_time = _ignore_value_
    problem = root_infeasible_max.RootInfeasibleMax()
    _execute_single_test(problem,
                         baseline,
                         solver=solver)
    for queue_strategy in sorted(pybnb.QueueStrategy):
        if queue_strategy == "custom":
            continue
        _execute_single_test(
            problem,
            baseline,
            solver=solver,
            queue_strategy=queue_strategy)

def _test_root_infeasible_min(comm):
    solver = Solver(comm=comm)
    baseline = SolverResults()
    baseline.solution_status = "infeasible"
    baseline.termination_condition = "optimality"
    baseline.objective = inf
    baseline.bound = inf
    baseline.nodes = 1
    baseline.wall_time = _ignore_value_
    problem = root_infeasible_min.RootInfeasibleMin()
    _execute_single_test(problem,
                         baseline,
                         solver=solver)
    for queue_strategy in sorted(pybnb.QueueStrategy):
        if queue_strategy == "custom":
            continue
        _execute_single_test(
            problem,
            baseline,
            solver=solver,
            queue_strategy=queue_strategy)

def _test_unbounded_max(comm):
    solver = Solver(comm=comm)
    baseline = SolverResults()
    baseline.solution_status = "unbounded"
    baseline.termination_condition = "no_nodes"
    baseline.objective = inf
    baseline.bound = inf
    baseline.nodes = 1
    baseline.wall_time = _ignore_value_
    problem = unbounded_max.UnboundedMax()
    _execute_single_test(problem,
                         baseline,
                         solver=solver)
    for queue_strategy in sorted(pybnb.QueueStrategy):
        if queue_strategy == "custom":
            continue
        _execute_single_test(
            problem,
            baseline,
            solver=solver,
            queue_strategy=queue_strategy)

def _test_unbounded_min(comm):
    solver = Solver(comm=comm)
    baseline = SolverResults()
    baseline.solution_status = "unbounded"
    baseline.termination_condition = "no_nodes"
    baseline.objective = -inf
    baseline.bound = -inf
    baseline.nodes = 1
    baseline.wall_time = _ignore_value_
    problem = unbounded_min.UnboundedMin()
    _execute_single_test(problem,
                         baseline,
                         solver=solver)
    for queue_strategy in sorted(pybnb.QueueStrategy):
        if queue_strategy == "custom":
            continue
        _execute_single_test(
            problem,
            baseline,
            solver=solver,
            queue_strategy=queue_strategy)

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
    baseline.termination_condition = "no_nodes"
    baseline.objective = 0.0
    baseline.bound = 0.0009765625
    baseline.absolute_gap = 0.0009765625
    baseline.relative_gap = 0.0009765625
    baseline.nodes = 2047
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
    baseline.wall_time = _ignore_value_
    problem = zero_objective_max.ZeroObjectiveMax()
    for queue_strategy in sorted(pybnb.QueueStrategy):
        if queue_strategy == "custom":
            continue
        if queue_strategy == "depth":
            baseline.nodes = 2033
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
    baseline.termination_condition = "no_nodes"
    baseline.objective = 0.0
    baseline.bound = -0.0009765625
    baseline.absolute_gap = 0.0009765625
    baseline.relative_gap = 0.0009765625
    baseline.nodes = 2047
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
    baseline.wall_time = _ignore_value_
    problem = zero_objective_min.ZeroObjectiveMin()
    for queue_strategy in sorted(pybnb.QueueStrategy):
        if queue_strategy == "custom":
            continue
        if queue_strategy == "depth":
            baseline.nodes = 2033
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
    baseline = SolverResults()
    baseline.solution_status = "unbounded"
    baseline.termination_condition = "no_nodes"
    baseline.objective = inf
    baseline.bound = inf
    baseline.nodes = _ignore_value_
    baseline.wall_time = _ignore_value_
    problem = delayed_unbounded_max.DelayedUnboundedMax()
    _execute_single_test(problem,
                         baseline,
                         solver=solver)
    for queue_strategy in sorted(pybnb.QueueStrategy):
        if queue_strategy == "custom":
            continue
        _execute_single_test(
            problem,
            baseline,
            solver=solver,
            queue_strategy=queue_strategy)

def _test_delayed_unbounded_min(comm):
    solver = Solver(comm=comm)
    baseline = SolverResults()
    baseline.solution_status = "unbounded"
    baseline.termination_condition = "no_nodes"
    baseline.objective = -inf
    baseline.bound = -inf
    baseline.nodes = _ignore_value_
    baseline.wall_time = _ignore_value_
    problem = delayed_unbounded_min.DelayedUnboundedMin()
    _execute_single_test(problem,
                         baseline,
                         solver=solver)
    for queue_strategy in sorted(pybnb.QueueStrategy):
        if queue_strategy == "custom":
            continue
        _execute_single_test(
            problem,
            baseline,
            solver=solver,
            queue_strategy=queue_strategy)

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
