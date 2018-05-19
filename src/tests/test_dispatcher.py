import pytest

from pybnb.common import (minimize,
                          inf)
from pybnb.misc import get_simple_logger
from pybnb.node import Node
from pybnb.convergence_checker import ConvergenceChecker
from pybnb.dispatcher import (Dispatcher,
                              DispatcherQueueData)
from pybnb.priority_queue import (WorstBoundFirstPriorityQueue,
                                  CustomPriorityQueue,
                                  BreadthFirstPriorityQueue,
                                  DepthFirstPriorityQueue)

class TestDispatcherSimple(object):

    def test_no_comm(self):
        with pytest.raises(ValueError):
            Dispatcher(None).serve()

    def test_node_priority_strategy(self):
        root = Node(size=0)
        Node._insert_tree_id(root._data, 0)
        initialize_queue = DispatcherQueueData(
            nodes=[root],
            next_tree_id=1)
        converger = ConvergenceChecker(minimize)
        node_limit = None
        time_limit = None
        log = get_simple_logger()
        log_interval_seconds = inf

        disp = Dispatcher(None)

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
