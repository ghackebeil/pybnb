import os
import tempfile
import time

import pytest

from pybnb.common import (minimize,
                          nan)
from pybnb.node import Node
from pybnb.problem import Problem
from pybnb.solver import (Solver,
                          summarize_worker_statistics,
                          solve)

from six import StringIO

class BadBranchSignatureProblem(Problem):
    def sense(self): return minimize
    def objective(self): return 0
    def bound(self): return 0
    def save_state(self, node): pass
    def load_state(self, node): pass
    def branch(self, node): raise NotImplementedError()

class DummyProblem(Problem):
    def sense(self): return minimize
    def objective(self): return 0
    def bound(self): return 0
    def save_state(self, node): pass
    def load_state(self, node): pass
    def branch(self): raise NotImplementedError()

class _DummyComm_Size1(object):
    size = 1

class TestSolverSimple(object):

    def test_bad_dispatcher_rank(self):
        with pytest.raises(ValueError):
            Solver(comm=None, dispatcher_rank=-1)
        with pytest.raises(ValueError):
            Solver(comm=None, dispatcher_rank=1)
        with pytest.raises(ValueError):
            Solver(comm=None, dispatcher_rank=1.1)
        Solver(comm=None, dispatcher_rank=0)
        Solver(comm=None)

    def test_no_mpi(self):
        b = Solver(comm=None)
        assert b.comm is None
        assert b.worker_comm is None
        assert b.is_worker == True
        assert b.is_dispatcher == True
        assert b.worker_count == 1
        b._reset_local_solve_stats()
        stats = b.collect_worker_statistics()
        assert len(stats) == 13
        assert stats['rank'] == [0]
        assert stats['wall_time'] == [0]
        assert stats['queue_time'] == [0]
        assert stats['queue_call_count'] == [0]
        assert stats['objective_time'] == [0]
        assert stats['objective_call_count'] == [0]
        assert stats['bound_time'] == [0]
        assert stats['bound_call_count'] == [0]
        assert stats['branch_time'] == [0]
        assert stats['branch_call_count'] == [0]
        assert stats['load_state_time'] == [0]
        assert stats['load_state_call_count'] == [0]
        assert stats['explored_nodes_count'] == [0]
        out = \
"""Number of Workers:        1
Load Imbalance:       0.00%
Average Worker Timing:
 - queue:       0.00% [avg time:   0.0 s , count: 0]
 - load_state:  0.00% [avg time:   0.0 s , count: 0]
 - bound:       0.00% [avg time:   0.0 s , count: 0]
 - objective:   0.00% [avg time:   0.0 s , count: 0]
 - branch:      0.00% [avg time:   0.0 s , count: 0]
 - other:       0.00% [avg time:   0.0 s , count: 0]
"""
        tmp = StringIO()
        summarize_worker_statistics(stats, stream=tmp)
        assert tmp.getvalue() == out

    def test_solve_function(self):
        fid, fname = tempfile.mkstemp()
        os.close(fid)
        try:
            solve(DummyProblem(),
                  comm=None,
                  log_filename=fname)
            assert os.path.exists(fname)
        finally:
            time.sleep(0.1)
            try:
                os.remove(fname)
            except:                               #pragma:nocover
                pass

    def test_bad_queue_strategy(self):
        with pytest.raises(ValueError):
            solve(DummyProblem(),
                  comm=None,
                  queue_strategy='_not_a_valid_strategy_')

    def test_bad_best_options(self):
        node = Node()
        node.objective = None
        with pytest.raises(ValueError):
            solve(DummyProblem(),
                  comm=None,
                  best_node=node)
        node.objective = nan
        with pytest.raises(ValueError):
            solve(DummyProblem(),
                  comm=None,
                  best_node=node)
        node.objective = 0
        solve(DummyProblem(),
              comm=None,
              best_node=node)

    def test_bad_branch_signature(self):
        problem = BadBranchSignatureProblem()
        with pytest.raises(TypeError):
            solve(problem,
                  comm=None)
