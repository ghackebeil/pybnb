import math

from pybnb.common import (minimize,
                          maximize,
                          inf,
                          nan,
                          QueueStrategy,
                          _queue_strategy_to_int,
                          _int_to_queue_strategy,
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

    def test_QueueStrategy(self):
        assert QueueStrategy.bound == "bound"
        assert QueueStrategy.objective == "objective"
        assert QueueStrategy.breadth == "breadth"
        assert QueueStrategy.depth == "depth"
        assert QueueStrategy.local_gap == "local_gap"
        assert QueueStrategy.fifo == "fifo"
        assert QueueStrategy.lifo == "lifo"
        assert QueueStrategy.random == "random"
        assert QueueStrategy.custom == "custom"
        assert len(QueueStrategy) == 9
        assert len(_queue_strategy_to_int) == 9
        assert len(_int_to_queue_strategy) == 9
        for val in QueueStrategy:
            assert _int_to_queue_strategy[
                _queue_strategy_to_int[val]] is val

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
        assert TerminationCondition.objective_limit == "objective_limit"
        assert TerminationCondition.node_limit == "node_limit"
        assert TerminationCondition.time_limit == "time_limit"
        assert TerminationCondition.queue_empty == "queue_empty"
        assert TerminationCondition.queue_limit == "queue_limit"
        assert TerminationCondition.interrupted == "interrupted"
        assert len(TerminationCondition) == 7
        assert len(_termination_condition_to_int) == 7
        assert len(_int_to_termination_condition) == 7
        for i, val in enumerate(TerminationCondition):
            assert _int_to_termination_condition[
                _termination_condition_to_int[val]] is val
