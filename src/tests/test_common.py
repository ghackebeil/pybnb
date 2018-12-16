import math

from pybnb.common import (minimize,
                          maximize,
                          inf,
                          nan,
                          NodePriorityStrategy,
                          _node_priority_strategy_to_int,
                          _int_to_node_priority_strategy,
                          SolutionStatus,
                          _solution_status_to_int,
                          _int_to_solution_status,
                          TerminationCondition,
                          _termination_condition_to_int,
                          _int_to_termination_condition)

class Test(object):

    def test_minimize(self):
        assert minimize == 1

    def test_maximize(self):
        assert maximize == -1

    def test_inf(self):
        assert math.isinf(inf)
        assert math.isinf(-inf)

    def test_nan(self):
        assert math.isnan(nan)
        assert math.isnan(-nan)

    def test_NodePriorityStrategy(self):
        assert NodePriorityStrategy.bound == "bound"
        assert NodePriorityStrategy.objective == "objective"
        assert NodePriorityStrategy.breadth == "breadth"
        assert NodePriorityStrategy.depth == "depth"
        assert NodePriorityStrategy.fifo == "fifo"
        assert NodePriorityStrategy.random == "random"
        assert NodePriorityStrategy.custom == "custom"
        assert len(NodePriorityStrategy) == 7
        assert len(_node_priority_strategy_to_int) == 7
        assert len(_int_to_node_priority_strategy) == 7
        for val in NodePriorityStrategy:
            assert _int_to_node_priority_strategy[
                _node_priority_strategy_to_int[val]] is val

    def test_SolutionStatus(self):
        assert SolutionStatus.optimal == "optimal"
        assert SolutionStatus.feasible == "feasible"
        assert SolutionStatus.infeasible == "infeasible"
        assert SolutionStatus.unbounded == "unbounded"
        assert SolutionStatus.invalid == "invalid"
        assert SolutionStatus.unknown == "unknown"
        assert len(SolutionStatus) == 6
        assert len(_solution_status_to_int) == 6
        assert len(_int_to_solution_status) == 6
        for i, val in enumerate(SolutionStatus):
            assert _int_to_solution_status[
                _solution_status_to_int[val]] is val

    def test_TerminationCondition(self):
        assert TerminationCondition.optimality == "optimality"
        assert TerminationCondition.cutoff == "cutoff"
        assert TerminationCondition.node_limit == "node_limit"
        assert TerminationCondition.time_limit == "time_limit"
        assert TerminationCondition.no_nodes == "no_nodes"
        assert TerminationCondition.interrupted == "interrupted"
        assert len(TerminationCondition) == 6
        assert len(_termination_condition_to_int) == 6
        assert len(_int_to_termination_condition) == 6
        for i, val in enumerate(TerminationCondition):
            assert _int_to_termination_condition[
                _termination_condition_to_int[val]] is val
