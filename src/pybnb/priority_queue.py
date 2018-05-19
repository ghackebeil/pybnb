"""
A collection of priority queue implementations that can be
used by the dispatcher.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""

import heapq

from pybnb.common import (minimize,
                          maximize)
from pybnb.node import Node

from sortedcontainers import SortedList

class _NoThreadingMaxPriorityFirstQueue(object):
    """A simple priority queue implementation that is not
    thread safe. When the queue is not empty, the item with
    the highest priority is next.

    This queue implementation is not allowed to store None.
    """

    def __init__(self):
        self._count = 0
        self._heap = []

    def size(self):
        """Returns the size of the queue."""
        return len(self._heap)

    def put(self, item, priority, _push_=heapq.heappush):
        """Puts an item into the queue with the given
        priority. Items placed in the queue may not be
        None. This method returns a unique counter associated
        with each put."""
        assert item is not None
        cnt = self._count
        self._count += 1
        _push_(self._heap, (-priority, cnt, item))
        return cnt

    def get(self, _pop_=heapq.heappop):
        """Removes and returns the highest priority item in
        the queue, where ties are broken by the order items
        were placed in the queue. If the queue is empty,
        returns None."""
        if len(self._heap) > 0:
            return _pop_(self._heap)[2]
        else:
            return None

    def next(self):
        """Returns, but does not remove, the highest
        priority item in the queue, where ties are broken by
        the order items were placed in the queue. If the
        queue is empty, returns None."""
        if len(self._heap) > 0:
            return self._heap[0][2]
        else:
            return None

    def filter(self, func, include_counters=False):
        """Removes items from the queue for which
        `func(item)` returns False. The list of items
        removed is returned. If `include_counters` is set to
        True, values in the returned list will have the form
        (cnt, item), where cnt is a unique counter that was
        created for the item when it was added to the
        queue."""
        heap_new = []
        removed = []
        for priority, cnt, item in self._heap:
            if func(item):
                heap_new.append((priority, cnt, item))
            elif not include_counters:
                removed.append(item)
            else:
                removed.append((cnt,item))
        heapq.heapify(heap_new)
        self._heap = heap_new
        return removed

    def items(self):
        """Iterates over the queued items in arbitrary order
        without modifying the queue."""
        for _,_,item in self._heap:
            yield item

class IPriorityQueue(object):
    """The abstract interface for priority queues that store
    node data for the dispatcher."""

    def size(self):                               #pragma:nocover
        """Returns the size of the queue."""
        raise NotImplementedError

    def put(self, data):                          #pragma:nocover
        """Puts a node data item in the queue, possibly
        updating the value of :attr:`queue_priority
        <pybnb.node.Node.queue_priority>`, depending on the
        queue implementation."""
        raise NotImplementedError()

    def get(self):                                #pragma:nocover
        """Returns the next data item in the queue. If the
        queue is empty, returns None."""
        raise NotImplementedError()

    def bound(self):                              #pragma:nocover
        """Returns the weakest bound of all data items in the
        queue. If the queue is empty, returns None."""
        raise NotImplementedError()

    def filter(self, func):                       #pragma:nocover
        """Removes items from the queue for which
        `func(item)` returns False. The list of items
        removed is returned. If the queue is empty or no
        items are removed, the returned list will be
        empty."""
        raise NotImplementedError()

    def items(self):                              #pragma:nocover
        """Iterates over the queued items in arbitrary order
        without modifying the queue."""
        raise NotImplementedError()

class WorstBoundFirstPriorityQueue(IPriorityQueue):
    """A priority queue implementation that returns the node
    data item with the worst bound first.

    Parameters
    ----------
    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    def __init__(self, sense):
        assert sense in (minimize, maximize)
        self._sense = sense
        self._queue = _NoThreadingMaxPriorityFirstQueue()

    def size(self):
        return self._queue.size()

    def put(self, data):
        bound = Node._extract_bound(data)
        if self._sense == minimize:
            priority = -bound
        else:
            priority = bound
        Node._insert_queue_priority(data, priority)
        self._queue.put(data, priority)

    def get(self):
        return self._queue.get()

    def bound(self):
        bound = None
        if self._queue.size() > 0:
            data = self._queue.next()
            bound = Node._extract_bound(data)
            priority = Node._extract_queue_priority(data)
            if self._sense == minimize:
                assert bound == -priority
            else:
                assert bound == priority
        return bound

    def filter(self, func):
        return self._queue.filter(func)

    def items(self):
        """Iterates over the queued items in arbitrary order
        without modifying the queue."""
        return self._queue.items()

class CustomPriorityQueue(IPriorityQueue):
    """A priority queue implementation that can handle
    custom node priorities. It uses an additional data
    structure to reduce the amount of time it takes to
    compute a queue bound.

    Parameters
    ----------
    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    def __init__(self, sense):
        assert sense in (minimize, maximize)
        self._sense = sense
        self._queue = _NoThreadingMaxPriorityFirstQueue()
        self._sorted_by_bound = SortedList()

    def size(self):
        return self._queue.size()

    def put(self, data):
        bound = Node._extract_bound(data)
        if not Node._has_queue_priority(data):
            raise ValueError("A node queue priority is required")
        priority = Node._extract_queue_priority(data)
        cnt = self._queue.put(data, priority)
        if self._sense == maximize:
            self._sorted_by_bound.add((-bound, cnt, data))
        else:
            self._sorted_by_bound.add((bound, cnt, data))

    def get(self):
        if self._queue.size() > 0:
            _,cnt,data_ = self._queue._heap[0]
            assert type(cnt) is int
            data = self._queue.get()
            assert data_ is data
            bound = Node._extract_bound(data)
            if self._sense == maximize:
                self._sorted_by_bound.remove((-bound, cnt, data))
            else:
                self._sorted_by_bound.remove((bound, cnt, data))
            return data
        else:
            return None

    def bound(self):
        assert self._queue.size() == len(self._sorted_by_bound)
        if self._queue.size() > 0:
            return Node._extract_bound(self._sorted_by_bound[0][2])
        else:
            return None

    def filter(self, func):
        removed = []
        for cnt, data in self._queue.filter(func,
                                            include_counters=True):
            removed.append(data)
            bound = Node._extract_bound(data)
            if self._sense == maximize:
                self._sorted_by_bound.remove((-bound, cnt, data))
            else:
                self._sorted_by_bound.remove((bound, cnt, data))
        return removed

    def items(self):
        """Iterates over the queued items in arbitrary order
        without modifying the queue."""
        return self._queue.items()

class BreadthFirstPriorityQueue(CustomPriorityQueue):
    """A priority queue implementation that serves nodes in
    breadth-first order.

    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    def put(self, data):
        depth = Node._extract_tree_depth(data)
        assert depth >= 0
        Node._insert_queue_priority(data, -depth)
        super(BreadthFirstPriorityQueue, self).put(data)

class DepthFirstPriorityQueue(CustomPriorityQueue):
    """A priority queue implementation that serves nodes in
    depth-first order.

    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    def put(self, data):
        depth = Node._extract_tree_depth(data)
        assert depth >= 0
        Node._insert_queue_priority(data, depth)
        super(DepthFirstPriorityQueue, self).put(data)
