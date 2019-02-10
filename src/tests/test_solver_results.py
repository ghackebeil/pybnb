import math

import pytest

from pybnb.common import (inf,
                          nan,
                          SolutionStatus,
                          TerminationCondition)
from pybnb.node import (Node, loads)
from pybnb.solver_results import SolverResults

from six import StringIO

yaml_available = False
try:
    import yaml
    yaml_available = True
except ImportError:
    pass

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
        results.solution_status = "optimal"
        results.termination_condition = "optimality"
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
        results.solution_status = "optimal"
        results.termination_condition = "optimality"
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
