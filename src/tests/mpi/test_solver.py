from pybnb.common import (minimize,
                          maximize,
                          inf)
from pybnb.node import Node
from pybnb.problem import Problem
from pybnb.solver import Solver
from pybnb.dispatcher import DispatcherQueueData

from .common import mpi_available

from runtests.mpi import MPITest

class DummyProblem(Problem):
    def __init__(self, sense): self._sense = sense
    def sense(self): return self._sense
    def objective(self): return 0
    def bound(self): return 0
    def save_state(self, node): pass
    def load_state(self, node): pass
    def branch(self, node): return ()

def _test_initialize_queue(comm):
    solver = Solver(comm=comm)

    # no initial queue
    for sense in (minimize, maximize):
        results = solver.solve(DummyProblem(sense))
        assert results.solution_status == "optimal"
        assert results.termination_condition == "optimality"
        assert results.objective == 0
        assert results.bound == 0
        assert results.absolute_gap == 0
        assert results.relative_gap == 0
        assert results.nodes == 1
        assert results.wall_time is not None
        assert not hasattr(results, "best_node")
        results = solver.solve(DummyProblem(sense),
                               best_objective=(1 if (sense == minimize) else -1))
        assert results.solution_status == "optimal"
        assert results.termination_condition == "optimality"
        assert results.objective == 0
        assert results.bound == 0
        assert results.absolute_gap == 0
        assert results.relative_gap == 0
        assert results.nodes == 1
        assert results.wall_time is not None
        assert not hasattr(results, "best_node")
        results = solver.solve(DummyProblem(sense),
                               best_objective=(1 if (sense == minimize) else -1),
                               disable_objective_call=True)
        assert results.solution_status == "feasible"
        assert results.termination_condition == "no_nodes"
        assert results.objective == (1 if (sense == minimize) else -1)
        assert results.bound == 0
        assert results.absolute_gap == 1
        assert results.relative_gap == 1
        assert results.nodes == 1
        assert results.wall_time is not None
        assert not hasattr(results, "best_node")

    # empty initial queue
    queue = DispatcherQueueData(
        nodes=[],
        next_tree_id=0)
    for sense in (minimize, maximize):
        results = solver.solve(DummyProblem(sense),
                               initialize_queue=queue)
        assert results.solution_status == "unknown"
        assert results.termination_condition == "no_nodes"
        assert results.objective == (inf if (sense == minimize) else -inf)
        assert results.bound == (-inf if (sense == minimize) else inf)
        assert results.absolute_gap is None
        assert results.relative_gap is None
        assert results.nodes == 0
        assert results.wall_time is not None
        assert not hasattr(results, "best_node")
        results = solver.solve(DummyProblem(sense),
                               initialize_queue=queue,
                               best_objective=0)
        assert results.solution_status == "feasible"
        assert results.termination_condition == "no_nodes"
        assert results.objective == 0
        assert results.bound == (-inf if (sense == minimize) else inf)
        assert results.absolute_gap == inf
        assert results.relative_gap == inf
        assert results.nodes == 0
        assert results.wall_time is not None
        assert not hasattr(results, "best_node")
        results = solver.solve(DummyProblem(sense),
                               initialize_queue=queue,
                               best_objective=0,
                               disable_objective_call=True)
        assert results.solution_status == "feasible"
        assert results.termination_condition == "no_nodes"
        assert results.objective == 0
        assert results.bound == (-inf if (sense == minimize) else inf)
        assert results.absolute_gap == inf
        assert results.relative_gap == inf
        assert results.nodes == 0
        assert results.wall_time is not None
        assert not hasattr(results, "best_node")

    # non-empty initial queue
    node = Node()
    node.tree_id = 0
    node.tree_depth = 0
    node.objective = 0
    node.bound = 0
    queue = DispatcherQueueData(
        nodes=[node],
        next_tree_id=1)
    orig_objective = queue.nodes[0].objective
    for sense in (minimize, maximize):
        queue.nodes[0].objective = orig_objective
        results = solver.solve(DummyProblem(sense),
                               initialize_queue=queue)
        assert results.solution_status == "optimal"
        assert results.termination_condition == "optimality"
        assert results.objective == 0
        assert results.bound == 0
        assert results.absolute_gap == 0
        assert results.relative_gap == 0
        assert results.nodes == 0
        assert results.wall_time is not None
        assert not hasattr(results, "best_node")
        queue.nodes[0].objective = orig_objective
        results = solver.solve(DummyProblem(sense),
                               initialize_queue=queue,
                               best_objective=(1 if (sense == minimize) else -1))
        assert results.solution_status == "optimal"
        assert results.termination_condition == "optimality"
        assert results.objective == 0
        assert results.bound == 0
        assert results.absolute_gap == 0
        assert results.relative_gap == 0
        assert results.nodes == 0
        assert results.wall_time is not None
        assert not hasattr(results, "best_node")
        queue.nodes[0].objective = (inf if (sense == minimize) else -inf)
        results = solver.solve(DummyProblem(sense),
                               initialize_queue=queue,
                               best_objective=(1 if (sense == minimize) else -1))
        assert results.solution_status == "optimal"
        assert results.termination_condition == "optimality"
        assert results.objective == 0
        assert results.bound == 0
        assert results.absolute_gap == 0
        assert results.relative_gap == 0
        assert results.nodes == 1
        assert results.wall_time is not None
        assert not hasattr(results, "best_node")
        queue.nodes[0].objective = (inf if (sense == minimize) else -inf)
        results = solver.solve(DummyProblem(sense),
                               initialize_queue=queue,
                               best_objective=(1 if (sense == minimize) else -1),
                               disable_objective_call=True)
        assert results.solution_status == "feasible"
        assert results.termination_condition == "no_nodes"
        assert results.objective == (1 if (sense == minimize) else -1)
        assert results.bound == 0
        assert results.absolute_gap == 1
        assert results.relative_gap == 1
        assert results.nodes == 1
        assert results.wall_time is not None
        assert not hasattr(results, "best_node")
        queue.nodes[0].objective = orig_objective

def test_initialize_queue_nocomm():
    _test_initialize_queue(None)

def test_solver_nocomm():
    solver = Solver(comm=None)
    assert solver.is_worker
    assert solver.is_dispatcher
    assert solver.comm is None
    assert solver.worker_comm is None

if mpi_available:

    @MPITest(commsize=[1, 2, 3])
    def test_initialize_queue(comm):
        _test_initialize_queue(comm)

    @MPITest(commsize=[1, 2, 3])
    def test_solver_comm(comm):
        solver = Solver(comm=comm, dispatcher_rank=0)
        if comm.size > 1:
            if comm.rank == 0:
                assert solver.is_dispatcher
                assert not solver.is_worker
                assert solver.comm is comm
                assert solver.worker_comm is None
            else:
                assert not solver.is_dispatcher
                assert solver.is_worker
                assert solver.comm is comm
                assert solver.worker_comm is not None
        else:
            assert solver.is_dispatcher
            assert solver.is_worker
            assert solver.comm is comm
            assert solver.worker_comm is comm
