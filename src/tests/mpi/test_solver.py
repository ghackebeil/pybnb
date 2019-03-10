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
    def __init__(self, sense):
        self._sense = sense
        self._notify_new_best_node_args = None
        self._notify_new_best_node_call_count = 0
    def sense(self): return self._sense
    def objective(self): return 0
    def bound(self): return 0
    def save_state(self, node): pass
    def load_state(self, node): pass
    def branch(self): return ()
    def notify_new_best_node(self, node, current):
        self._notify_new_best_node_args = (node, current)
        self._notify_new_best_node_call_count += 1

def _test_initialize_queue(comm):
    solver = Solver(comm=comm)

    # no initial queue
    for sense in (minimize, maximize):
        problem = DummyProblem(sense)
        results = solver.solve(problem,
                               queue_limit=0)
        assert results.solution_status == "unknown"
        assert results.termination_condition == "queue_limit"
        assert results.objective == (inf if (sense == minimize) else -inf)
        assert results.bound == (-inf if (sense == minimize) else inf)
        assert results.absolute_gap is None
        assert results.relative_gap is None
        assert results.nodes == 0
        assert results.wall_time is not None
        assert results.best_node is None
        assert problem._notify_new_best_node_call_count == 0
        assert solver._local_solve_info.explored_nodes_count == 0
        problem = DummyProblem(sense)
        results = solver.solve(problem)
        assert results.solution_status == "optimal"
        assert results.termination_condition == "optimality"
        assert results.objective == 0
        assert results.bound == 0
        assert results.absolute_gap == 0
        assert results.relative_gap == 0
        assert results.nodes == 1
        assert results.wall_time is not None
        assert results.best_node is not None
        assert results.best_node.objective == results.objective
        assert results.best_node.tree_depth == 0
        if solver.is_worker:
            assert problem._notify_new_best_node_call_count == 1
            assert problem._notify_new_best_node_args[0] is results.best_node
            if solver._local_solve_info.explored_nodes_count == 1:
                assert problem._notify_new_best_node_args[1]
            else:
                assert solver._local_solve_info.explored_nodes_count == 0
                assert not problem._notify_new_best_node_args[1]
        else:
            assert problem._notify_new_best_node_call_count == 0
        problem = DummyProblem(sense)
        results = solver.solve(problem,
                               best_objective=(1 if (sense == minimize) else -1))
        assert results.solution_status == "optimal"
        assert results.termination_condition == "optimality"
        assert results.objective == 0
        assert results.bound == 0
        assert results.absolute_gap == 0
        assert results.relative_gap == 0
        assert results.nodes == 1
        assert results.wall_time is not None
        assert results.best_node is not None
        assert results.best_node.objective == results.objective
        assert results.best_node.tree_depth == 0
        if solver.is_worker:
            assert problem._notify_new_best_node_call_count == 1
            assert problem._notify_new_best_node_args[0] is results.best_node
            if solver._local_solve_info.explored_nodes_count == 1:
                assert problem._notify_new_best_node_args[1]
            else:
                assert solver._local_solve_info.explored_nodes_count == 0
                assert not problem._notify_new_best_node_args[1]
        else:
            assert problem._notify_new_best_node_call_count == 0
        problem = DummyProblem(sense)
        results = solver.solve(problem,
                               best_objective=(1 if (sense == minimize) else -1),
                               disable_objective_call=True)
        assert results.solution_status == "feasible"
        assert results.termination_condition == "queue_empty"
        assert results.objective == (1 if (sense == minimize) else -1)
        assert results.bound == 0
        assert results.absolute_gap == 1
        assert results.relative_gap == 1
        assert results.nodes == 1
        assert results.wall_time is not None
        assert results.best_node is None
        assert problem._notify_new_best_node_call_count == 0
        best_node_ = Node()
        best_node_.objective = (1 if (sense == minimize) else -1)
        best_node_._uuid = 'abcd'
        problem = DummyProblem(sense)
        results = solver.solve(problem,
                               best_node=best_node_,
                               disable_objective_call=True)
        assert results.solution_status == "feasible"
        assert results.termination_condition == "queue_empty"
        assert results.objective == best_node_.objective
        assert results.bound == 0
        assert results.absolute_gap == 1
        assert results.relative_gap == 1
        assert results.nodes == 1
        assert results.best_node._uuid == best_node_._uuid
        if (comm is None) or (comm.size == 1):
            assert results.best_node is best_node_
        assert results.best_node.objective == results.objective
        if solver.is_worker:
            assert problem._notify_new_best_node_call_count == 1
            assert problem._notify_new_best_node_args[0] is results.best_node
            if solver._local_solve_info.explored_nodes_count == 1:
                assert not problem._notify_new_best_node_args[1]
            else:
                assert solver._local_solve_info.explored_nodes_count == 0
                assert not problem._notify_new_best_node_args[1]
        else:
            assert problem._notify_new_best_node_call_count == 0
        best_node_ = Node()
        best_node_.objective = (1 if (sense == minimize) else -1)
        best_node_._uuid = 'abcd'
        problem = DummyProblem(sense)
        results = solver.solve(problem,
                               best_node=best_node_)
        assert results.solution_status == "optimal"
        assert results.termination_condition == "optimality"
        assert results.objective == 0
        assert results.bound == 0
        assert results.absolute_gap == 0
        assert results.relative_gap == 0
        assert results.nodes == 1
        assert results.wall_time is not None
        assert results.best_node._uuid != best_node_._uuid
        if (comm is None) or (comm.size == 1):
            assert results.best_node is not best_node_
        assert results.best_node.objective == results.objective
        assert results.best_node.tree_depth == 0
        if solver.is_worker:
            assert problem._notify_new_best_node_call_count >= 1
            assert problem._notify_new_best_node_args[0] is results.best_node
            if problem._notify_new_best_node_call_count == 2:
                assert problem._notify_new_best_node_args[1]
            else:
                assert problem._notify_new_best_node_args[0]
        else:
            assert problem._notify_new_best_node_call_count == 0

    # empty initial queue
    queue = DispatcherQueueData(
        nodes=[],
        worst_terminal_bound=None,
        sense=minimize)
    for sense in (minimize, maximize):
        queue.sense = sense
        problem = DummyProblem(sense)
        results = solver.solve(problem,
                               initialize_queue=queue)
        assert results.solution_status == "unknown"
        assert results.termination_condition == "queue_empty"
        assert results.objective == (inf if (sense == minimize) else -inf)
        assert results.bound == (-inf if (sense == minimize) else inf)
        assert results.absolute_gap is None
        assert results.relative_gap is None
        assert results.nodes == 0
        assert results.wall_time is not None
        assert results.best_node is None
        assert problem._notify_new_best_node_call_count == 0
        problem = DummyProblem(sense)
        results = solver.solve(problem,
                               initialize_queue=queue,
                               best_objective=0)
        assert results.solution_status == "feasible"
        assert results.termination_condition == "queue_empty"
        assert results.objective == 0
        assert results.bound == (-inf if (sense == minimize) else inf)
        assert results.absolute_gap == inf
        assert results.relative_gap == inf
        assert results.nodes == 0
        assert results.wall_time is not None
        assert results.best_node is None
        assert problem._notify_new_best_node_call_count == 0
        problem = DummyProblem(sense)
        results = solver.solve(problem,
                               initialize_queue=queue,
                               best_objective=0,
                               disable_objective_call=True)
        assert results.solution_status == "feasible"
        assert results.termination_condition == "queue_empty"
        assert results.objective == 0
        assert results.bound == (-inf if (sense == minimize) else inf)
        assert results.absolute_gap == inf
        assert results.relative_gap == inf
        assert results.nodes == 0
        assert results.wall_time is not None
        assert results.best_node is None
        assert problem._notify_new_best_node_call_count == 0
        best_node_ = Node()
        best_node_.objective = (1 if (sense == minimize) else -1)
        best_node_._uuid = 'abcd'
        problem = DummyProblem(sense)
        results = solver.solve(problem,
                               initialize_queue=queue,
                               best_objective=0,
                               best_node=best_node_)
        assert results.solution_status == "feasible"
        assert results.termination_condition == "queue_empty"
        assert results.objective == 0
        assert results.bound == (-inf if (sense == minimize) else inf)
        assert results.absolute_gap == inf
        assert results.relative_gap == inf
        assert results.nodes == 0
        assert results.wall_time is not None
        assert results.best_node._uuid == best_node_._uuid
        if (comm is None) or (comm.size == 1):
            assert results.best_node is best_node_
        if solver.is_worker:
            assert problem._notify_new_best_node_call_count == 1
            assert problem._notify_new_best_node_args[0] is results.best_node
            assert solver._local_solve_info.explored_nodes_count == 0
            assert not problem._notify_new_best_node_args[1]
        else:
            assert problem._notify_new_best_node_call_count == 0
        best_node_ = Node()
        best_node_.objective = (1 if (sense == minimize) else -1)
        best_node_._uuid = 'abcd'
        problem = DummyProblem(sense)
        results = solver.solve(problem,
                               initialize_queue=queue,
                               best_objective=(2 if (sense == minimize) else -2),
                               best_node=best_node_)
        assert results.solution_status == "feasible"
        assert results.termination_condition == "queue_empty"
        assert results.objective == (1 if (sense == minimize) else -1)
        assert results.bound == (-inf if (sense == minimize) else inf)
        assert results.absolute_gap == inf
        assert results.relative_gap == inf
        assert results.nodes == 0
        assert results.wall_time is not None
        assert results.best_node._uuid == best_node_._uuid
        if (comm is None) or (comm.size == 1):
            assert results.best_node is best_node_
        if solver.is_worker:
            assert problem._notify_new_best_node_call_count == 1
            assert problem._notify_new_best_node_args[0] is results.best_node
            assert solver._local_solve_info.explored_nodes_count == 0
            assert not problem._notify_new_best_node_args[1]
        else:
            assert problem._notify_new_best_node_call_count == 0

    # non-empty initial queue
    root = Node()
    root._uuid = 'abcd'
    root.tree_depth = 0
    root.objective = 0
    root.bound = 0
    queue = DispatcherQueueData(
        nodes=[root],
        worst_terminal_bound=None,
        sense=minimize)
    orig_objective = queue.nodes[0].objective
    for sense in (minimize, maximize):
        queue.sense = sense
        queue.nodes[0].objective = orig_objective
        problem = DummyProblem(sense)
        results = solver.solve(problem,
                               initialize_queue=queue)
        assert results.solution_status == "optimal"
        assert results.termination_condition == "optimality"
        assert results.objective == 0
        assert results.bound == 0
        assert results.absolute_gap == 0
        assert results.relative_gap == 0
        assert results.nodes == 1
        assert results.wall_time is not None
        assert results.best_node._uuid == root._uuid
        if (comm is None) or (comm.size == 1):
            assert results.best_node is root
        if solver.is_worker:
            assert problem._notify_new_best_node_call_count == 1
            assert problem._notify_new_best_node_args[0] is results.best_node
            if solver._local_solve_info.explored_nodes_count == 1:
                assert problem._notify_new_best_node_args[1]
            else:
                assert solver._local_solve_info.explored_nodes_count == 0
                assert not problem._notify_new_best_node_args[1]
        else:
            assert problem._notify_new_best_node_call_count == 0
        queue.nodes[0].objective = orig_objective
        problem = DummyProblem(sense)
        results = solver.solve(problem,
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
        assert results.best_node._uuid == root._uuid
        if (comm is None) or (comm.size == 1):
            assert results.best_node is root
        if solver.is_worker:
            assert problem._notify_new_best_node_call_count == 1
            assert problem._notify_new_best_node_args[0] is results.best_node
            if solver._local_solve_info.explored_nodes_count == 1:
                assert problem._notify_new_best_node_args[1]
            else:
                assert solver._local_solve_info.explored_nodes_count == 0
                assert not problem._notify_new_best_node_args[1]
        else:
            assert problem._notify_new_best_node_call_count == 0
        queue.nodes[0].objective = (inf if (sense == minimize) else -inf)
        problem = DummyProblem(sense)
        results = solver.solve(problem,
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
        assert results.best_node._uuid == root._uuid
        if (comm is None) or (comm.size == 1):
            assert results.best_node is root
        if solver.is_worker:
            assert problem._notify_new_best_node_call_count == 1
            assert problem._notify_new_best_node_args[0] is results.best_node
            if solver._local_solve_info.explored_nodes_count == 1:
                assert problem._notify_new_best_node_args[1]
            else:
                assert solver._local_solve_info.explored_nodes_count == 0
                assert not problem._notify_new_best_node_args[1]
        else:
            assert problem._notify_new_best_node_call_count == 0
        queue.nodes[0].objective = (inf if (sense == minimize) else -inf)
        problem = DummyProblem(sense)
        results = solver.solve(problem,
                               initialize_queue=queue,
                               best_objective=(1 if (sense == minimize) else -1),
                               disable_objective_call=True)
        assert results.solution_status == "feasible"
        assert results.termination_condition == "queue_empty"
        assert results.objective == (1 if (sense == minimize) else -1)
        assert results.bound == 0
        assert results.absolute_gap == 1
        assert results.relative_gap == 1
        assert results.nodes == 1
        assert results.wall_time is not None
        assert results.best_node is None
        assert problem._notify_new_best_node_call_count == 0
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
