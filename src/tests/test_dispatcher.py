import pytest

from pybnb.common import (minimize,
                          inf)
from pybnb.misc import get_simple_logger
from pybnb.node import Node
from pybnb.convergence_checker import ConvergenceChecker
from pybnb.dispatcher import (DispatcherLocal,
                              DispatcherQueueData)
from pybnb.priority_queue import (WorstBoundFirstPriorityQueue,
                                  CustomPriorityQueue,
                                  BestObjectiveFirstPriorityQueue,
                                  BreadthFirstPriorityQueue,
                                  DepthFirstPriorityQueue,
                                  FIFOQueue,
                                  RandomPriorityQueue,
                                  LocalGapPriorityQueue)

class TestDispatcherSimple(object):

    def test_queue_strategy(self):
        converger = ConvergenceChecker(minimize)
        root = Node(size=0)
        Node._insert_tree_id(root._data, 0)
        Node._insert_bound(root._data,
                           converger.unbounded_objective)
        Node._insert_objective(root._data,
                               converger.infeasible_objective)
        initialize_queue = DispatcherQueueData(
            nodes=[root],
            next_tree_id=1)

        node_limit = None
        time_limit = None
        log = get_simple_logger()
        log_interval_seconds = inf

        disp = DispatcherLocal()

        disp.initialize(
            inf,
            initialize_queue,
            'bound',
            converger,
            node_limit,
            time_limit,
            log,
            log_interval_seconds)
        assert type(disp.queue) is WorstBoundFirstPriorityQueue
        disp.initialize(
            inf,
            initialize_queue,
            'custom',
            converger,
            node_limit,
            time_limit,
            log,
            log_interval_seconds)
        assert type(disp.queue) is CustomPriorityQueue
        disp.initialize(
            inf,
            initialize_queue,
            'objective',
            converger,
            node_limit,
            time_limit,
            log,
            log_interval_seconds)
        assert type(disp.queue) is BestObjectiveFirstPriorityQueue
        disp.initialize(
            inf,
            initialize_queue,
            'breadth',
            converger,
            node_limit,
            time_limit,
            log,
            log_interval_seconds)
        assert type(disp.queue) is BreadthFirstPriorityQueue
        disp.initialize(
            inf,
            initialize_queue,
            'depth',
            converger,
            node_limit,
            time_limit,
            log,
            log_interval_seconds)
        assert type(disp.queue) is DepthFirstPriorityQueue
        disp.initialize(
            inf,
            initialize_queue,
            'fifo',
            converger,
            node_limit,
            time_limit,
            log,
            log_interval_seconds)
        assert type(disp.queue) is FIFOQueue
        disp.initialize(
            inf,
            initialize_queue,
            'random',
            converger,
            node_limit,
            time_limit,
            log,
            log_interval_seconds)
        assert type(disp.queue) is RandomPriorityQueue
        disp.initialize(
            inf,
            initialize_queue,
            'local_gap',
            converger,
            node_limit,
            time_limit,
            log,
            log_interval_seconds)
        assert type(disp.queue) is LocalGapPriorityQueue
