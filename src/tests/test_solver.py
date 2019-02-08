import os
import tempfile
import time
import math

import pytest

from pybnb.common import (minimize,
                          inf,
                          nan,
                          SolutionStatus,
                          TerminationCondition)
from pybnb.node import (Node, loads)
from pybnb.problem import Problem
from pybnb.solver_results import SolverResults
from pybnb.solver import (Solver,
                          summarize_worker_statistics,
                          solve)

from six import StringIO

yaml_available = False
try:
    import yaml
    yaml_available = True
except ImportError:
    pass

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

class TestSolverResults(object):

    def test_pprint(self):
        results = SolverResults()
        out = StringIO()
        results.pprint(stream=out)
        assert out.getvalue() == \
            """solver results:
 - solution_status: None
 - termination_condition: None
 - objective: None
 - bound: None
 - absolute_gap: None
 - relative_gap: None
 - nodes: None
 - wall_time: None
 - best_node: None
"""
        results.best_node = Node()
        out = StringIO()
        results.pprint(stream=out)
        assert out.getvalue() == \
            """solver results:
 - solution_status: None
 - termination_condition: None
 - objective: None
 - bound: None
 - absolute_gap: None
 - relative_gap: None
 - nodes: None
 - wall_time: None
 - best_node: Node(objective=None)
"""
        results.solution_status = \
            SolutionStatus.optimal
        results.termination_condition = \
            TerminationCondition.optimality
        results.objective = 1
        results.bound = 0
        results.absolute_gap = 1
        results.relative_gap = 1
        results.nodes = 1
        results.wall_time = 60
        results.best_node = Node()
        results.best_node.tree_depth = 3
        results.best_node.objective = 4
        results.junk0 = 1.0
        results.junk1 = 1
        results.junk2 = inf
        results.junk3 = -inf
        results.junk4 = nan
        results.junk5 = 'inf'
        results.junk6 = '-inf'
        results.junk7 = 'nan'
        results.junk8 = None
        results.junk9 = 'None'
        out = StringIO()
        results.pprint(stream=out)
        assert out.getvalue() == \
            """solver results:
 - solution_status: optimal
 - termination_condition: optimality
 - objective: 1
 - bound: 0
 - absolute_gap: 1
 - relative_gap: 1
 - nodes: 1
 - wall_time: 1.00 m
 - best_node: Node(objective=4)
 - junk0: 1.0
 - junk1: 1
 - junk2: inf
 - junk3: -inf
 - junk4: nan
 - junk5: 'inf'
 - junk6: '-inf'
 - junk7: 'nan'
 - junk8: None
 - junk9: 'None'
"""

    def test_write(self):
        if not yaml_available:
            pytest.skip("yaml is not available")
        results = SolverResults()
        out = StringIO()
        results.write(out)
        x = yaml.load(out.getvalue())
        assert len(x) == 9
        assert x['solution_status'] is None
        assert x['termination_condition'] is None
        assert x['objective'] is None
        assert x['bound'] is None
        assert x['absolute_gap'] is None
        assert x['relative_gap'] is None
        assert x['nodes'] is None
        assert x['wall_time'] is None
        assert x['best_node'] is None
        results.solution_status = \
            SolutionStatus.optimal
        results.termination_condition = \
            TerminationCondition.optimality
        del results.objective
        results.bound = inf
        results.absolute_gap = -inf
        results.relative_gap = nan
        results.nodes = 1
        results.wall_time = 1.0
        results.best_node = Node()
        results.best_node.queue_priority = 12
        results.best_node.tree_depth = 3
        results.best_node.state = (1,2)
        results.best_node.objective = 1.5
        results.best_node.bound = -1.4
        results.junk0 = 1.0
        results.junk1 = 1
        results.junk2 = inf
        results.junk3 = -inf
        results.junk4 = nan
        results.junk5 = 'inf'
        results.junk6 = '-inf'
        results.junk7 = 'nan'
        out = StringIO()
        results.write(out)
        x = yaml.load(out.getvalue())
        assert len(x) == 16
        assert x['solution_status'] == "optimal"
        assert x['termination_condition'] == "optimality"
        assert x['bound'] == inf
        assert x['absolute_gap'] == -inf
        assert math.isnan(x['relative_gap'])
        assert x['nodes'] == 1
        assert type(x['nodes']) is int
        assert x['wall_time'] == 1.0
        assert type(x['wall_time']) is float
        best_node_ = loads(x['best_node'])
        assert best_node_.queue_priority == 12
        assert best_node_.tree_depth == 3
        assert best_node_.state == (1,2)
        assert best_node_.objective == 1.5
        assert best_node_.bound == -1.4
        assert x['junk0'] == 1.0
        assert type(x['junk0']) is float
        assert x['junk1'] == 1
        assert type(x['junk1']) is int
        assert x['junk2'] == inf
        assert x['junk3'] == -inf
        assert math.isnan(x['junk4'])
        assert x['junk5'] == 'inf'
        assert x['junk6'] == '-inf'
        assert x['junk7'] == 'nan'

        results.bound = 'inf'
        results.absolute_gap = '-inf'
        results.relative_gap = 'nan'
        results.nodes = None
        results.junk0 = None
        out = StringIO()
        results.write(out)
        x = yaml.load(out.getvalue())
        assert len(x) == 16
        assert x['bound'] == 'inf'
        assert x['absolute_gap'] == '-inf'
        assert x['relative_gap'] == 'nan'
        assert x['nodes'] is None
        assert x['junk0'] is None

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
        assert len(stats) == 12
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
