"""
A collection of priority queue implementations that can be
used by the dispatcher.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""

import random
import collections
import heapq
import math

from pybnb.common import (minimize,
                          maximize)

from sortedcontainers import SortedList

class _NoThreadingMaxPriorityFirstQueue(object):
    """A simple priority queue implementation that is not
    thread safe. When the queue is not empty, the item with
    the highest priority is next.

    This queue implementation is not allowed to store None.
    """
    requires_priority = True

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
        assert not math.isnan(priority)
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

    def put_get(self, item, priority, _push_pop_=heapq.heappushpop):
        """Combines a put and get call, which can be more
        efficient than two separate put and get
        calls. Returns a tuple containing the put and get
        return values."""
        assert item is not None
        assert not math.isnan(priority)
        cnt = self._count
        self._count += 1
        if len(self._heap) > 0:
            return cnt, _push_pop_(self._heap, (-priority, cnt, item))[2]
        else:
            return cnt, item

    def next(self):
        """Returns, without modifying the queue, a tuple of
        the form (cnt, item), where item is highest priority
        entry in the queue and cnt is the unique counter
        assigned to it when it was added to the queue.

        Raises
        ------
        IndexError
            If the queue is empty.
        """
        try:
            return self._heap[0][1:]
        except IndexError:
            raise IndexError("The queue is empty")

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

class _NoThreadingFIFOQueue(object):
    """A simple first-in, first-out queue implementation
    that is not thread safe.

    This queue implementation is not allowed to store None.
    """
    requires_priority = False

    def __init__(self):
        self._count = 0
        self._deque = collections.deque()

    def size(self):
        """Returns the size of the queue."""
        return len(self._deque)

    def put(self, item):
        """Puts an item into the queue. Items placed in the
        queue may not be None. This method returns a unique
        counter associated with each put."""
        assert item is not None
        cnt = self._count
        self._count += 1
        self._deque.append((cnt, item))
        return cnt

    def get(self):
        """Removes and returns the next item in the
        queue. If the queue is empty, returns None."""
        if len(self._deque) > 0:
            return self._deque.popleft()[1]
        else:
            return None

    def put_get(self, item):
        """Combines a put and get call, which can be more
        efficient than two separate put and get
        calls. Returns a tuple containing the put and get
        return values."""
        assert item is not None
        cnt = self._count
        self._count += 1
        if len(self._deque) > 0:
            self._deque.rotate(-1)
            return_item = self._deque[-1][1]
            self._deque[-1] = (cnt, item)
            return cnt, return_item
        else:
            return cnt, item

    def next(self):
        """Returns, without modifying the queue, a tuple of
        the form (cnt, item), where item is highest priority
        entry in the queue and cnt is the unique counter
        assigned to it when it was added to the queue.

        Raises
        ------
        IndexError
            If the queue is empty.
        """
        try:
            return self._deque[0]
        except IndexError:
            raise IndexError("The queue is empty")

    def filter(self, func, include_counters=False):
        """Removes items from the queue for which
        `func(item)` returns False. The list of items
        removed is returned. If `include_counters` is set to
        True, values in the returned list will have the form
        (cnt, item), where cnt is a unique counter that was
        created for the item when it was added to the
        queue."""
        deque_new = collections.deque()
        removed = []
        for cnt, item in self._deque:
            if func(item):
                deque_new.append((cnt, item))
            elif not include_counters:
                removed.append(item)
            else:
                removed.append((cnt,item))
        self._deque = deque_new
        return removed

    def items(self):
        """Iterates over the queued items in arbitrary order
        without modifying the queue."""
        for _,item in self._deque:
            yield item

class _NoThreadingLIFOQueue(object):
    """A simple last-in, first-out queue implementation
    that is not thread safe.

    This queue implementation is not allowed to store None.
    """
    requires_priority = False

    def __init__(self):
        self._count = 0
        self._items = []

    def size(self):
        """Returns the size of the queue."""
        return len(self._items)

    def put(self, item):
        """Puts an item into the queue. Items placed in the
        queue may not be None. This method returns a unique
        counter associated with each put."""
        assert item is not None
        cnt = self._count
        self._count += 1
        self._items.append((cnt, item))
        return cnt

    def get(self):
        """Removes and returns the next item in the
        queue. If the queue is empty, returns None."""
        if len(self._items) > 0:
            return self._items.pop()[1]
        else:
            return None

    def put_get(self, item):
        """Combines a put and get call, which can be more
        efficient than two separate put and get
        calls. Returns a tuple containing the put and get
        return values."""
        assert item is not None
        cnt = self._count
        self._count += 1
        return cnt, item

    def next(self):
        """Returns, without modifying the queue, a tuple of
        the form (cnt, item), where item is highest priority
        entry in the queue and cnt is the unique counter
        assigned to it when it was added to the queue.

        Raises
        ------
        IndexError
            If the queue is empty.
        """
        try:
            return self._items[-1]
        except IndexError:
            raise IndexError("The queue is empty")

    def filter(self, func, include_counters=False):
        """Removes items from the queue for which
        `func(item)` returns False. The list of items
        removed is returned. If `include_counters` is set to
        True, values in the returned list will have the form
        (cnt, item), where cnt is a unique counter that was
        created for the item when it was added to the
        queue."""
        items_new = []
        removed = []
        for cnt, item in self._items:
            if func(item):
                items_new.append((cnt, item))
            elif not include_counters:
                removed.append(item)
            else:
                removed.append((cnt,item))
        self._items = items_new
        return removed

    def items(self):
        """Iterates over the queued items in arbitrary order
        without modifying the queue."""
        for _,item in self._items:
            yield item

class IPriorityQueue(object):
    """The abstract interface for priority queues that store
    node data for the dispatcher."""

    def size(self):                               #pragma:nocover
        """Returns the size of the queue."""
        raise NotImplementedError

    def put(self, node):                          #pragma:nocover
        """Puts an node in the queue, possibly updating the
        value of :attr:`queue_priority <pybnb.node.Node.queue_priority>`,
        depending on the queue implementation. This method
        returns a unique counter associated with each
        put."""
        raise NotImplementedError()

    def get(self):                                #pragma:nocover
        """Returns the next node in the queue. If the queue
        is empty, returns None."""
        raise NotImplementedError()

    def put_get(self, node):                          #pragma:nocover
        """Combines a put and get call, which can be more
        efficient than two separate put and get
        calls. Returns a tuple containing the put and get
        return values."""
        raise NotImplementedError()

    def bound(self):                              #pragma:nocover
        """Returns the weakest bound of all nodes in the
        queue. If the queue is empty, returns None."""
        raise NotImplementedError()

    def filter(self, func):                       #pragma:nocover
        """Removes nodes from the queue for which
        `func(node)` returns False. The list of nodes
        removed is returned. If the queue is empty or no
        nodes are removed, the returned list will be
        empty."""
        raise NotImplementedError()

    def items(self):                              #pragma:nocover
        """Iterates over the queued nodes in arbitrary order
        without modifying the queue."""
        raise NotImplementedError()

class WorstBoundFirstPriorityQueue(IPriorityQueue):
    """A priority queue implementation that serves nodes
    with the worst bound first.

    Parameters
    ----------
    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    def __init__(self, sense):
        assert sense in (minimize, maximize)
        self._sense = sense
        self._queue = _NoThreadingMaxPriorityFirstQueue()

    @staticmethod
    def generate_priority(node, sense):
        bound = node.bound
        if sense == minimize:
            return -bound
        else:
            assert sense == maximize
            return bound

    def size(self):
        return self._queue.size()

    def put(self, node):
        node.queue_priority = self.generate_priority(node, self._sense)
        return self._queue.put(node, node.queue_priority)

    def get(self):
        return self._queue.get()

    def put_get(self, node):
        node.queue_priority = self.generate_priority(node, self._sense)
        return self._queue.put_get(node, node.queue_priority)

    def bound(self):
        try:
            return self._queue.next()[1].bound
        except IndexError:
            return None

    def filter(self, func):
        return self._queue.filter(func)

    def items(self):
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

    def __init__(self,
                 sense,
                 _queue_type_=_NoThreadingMaxPriorityFirstQueue):
        assert sense in (minimize, maximize)
        self._sense = sense
        self._queue = _queue_type_()
        self._sorted_by_bound = SortedList()

    def size(self):
        return self._queue.size()

    def put(self, node):
        bound = node.bound
        assert not math.isnan(bound)
        if self._queue.requires_priority:
            priority = node.queue_priority
            if priority is None:
                raise ValueError("A node queue priority is required")
            cnt = self._queue.put(node, priority)
        else:
            cnt = self._queue.put(node)
        if self._sense == maximize:
            self._sorted_by_bound.add((-bound, cnt, node))
        else:
            self._sorted_by_bound.add((bound, cnt, node))
        return cnt

    def get(self):
        if self._queue.size() > 0:
            cnt, tmp_ = self._queue.next()
            assert type(cnt) is int
            node = self._queue.get()
            assert tmp_ is node
            bound = node.bound
            if self._sense == maximize:
                self._sorted_by_bound.remove((-bound, cnt, node))
            else:
                self._sorted_by_bound.remove((bound, cnt, node))
            return node
        else:
            return None

    def put_get(self, node):
        if self._queue.size() > 0:
            cnt_next, tmp_ = self._queue.next()
            assert type(cnt_next) is int
            if self._queue.requires_priority:
                priority = node.queue_priority
                if priority is None:
                    raise ValueError("A node queue priority is required")
                cnt, node_ = self._queue.put_get(node, priority)
            else:
                cnt, node_ = self._queue.put_get(node)
            if node_ is not node:
                assert node_ is tmp_
                bound = node.bound
                assert not math.isnan(bound)
                bound_ = node_.bound
                assert not math.isnan(bound_)
                if self._sense == maximize:
                    self._sorted_by_bound.add((-bound, cnt, node))
                    self._sorted_by_bound.remove((-bound_, cnt_next, node_))
                else:
                    self._sorted_by_bound.add((bound, cnt, node))
                    self._sorted_by_bound.remove((bound_, cnt_next, node_))
        else:
            if self._queue.requires_priority:
                priority = node.queue_priority
                if priority is None:
                    raise ValueError("A node queue priority is required")
                cnt, node_ = self._queue.put_get(node, priority)
            else:
                cnt, node_ = self._queue.put_get(node)

        return cnt, node_

    def bound(self):
        try:
            return self._sorted_by_bound[0][2].bound
        except IndexError:
            return None

    def filter(self, func):
        removed = []
        for cnt, node in self._queue.filter(func,
                                            include_counters=True):
            removed.append(node)
            bound = node.bound
            if self._sense == maximize:
                self._sorted_by_bound.remove((-bound, cnt, node))
            else:
                self._sorted_by_bound.remove((bound, cnt, node))
        return removed

    def items(self):
        return self._queue.items()

class BestObjectiveFirstPriorityQueue(CustomPriorityQueue):
    """A priority queue implementation that serves nodes
    with the best objective first.

    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    @staticmethod
    def generate_priority(node, sense):
        assert sense in (minimize, maximize)
        objective = node.objective
        if sense == minimize:
            return -objective
        else:
            assert sense == maximize
            return objective

    def put(self, node):
        node.queue_priority = self.generate_priority(node, self._sense)
        return super(BestObjectiveFirstPriorityQueue, self).put(node)

    def put_get(self, node):
        node.queue_priority = self.generate_priority(node, self._sense)
        return super(BestObjectiveFirstPriorityQueue, self).put_get(node)

class BreadthFirstPriorityQueue(CustomPriorityQueue):
    """A priority queue implementation that serves nodes in
    breadth-first order.

    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    @staticmethod
    def generate_priority(node, sense):
        assert sense in (minimize, maximize)
        assert node.tree_depth >= 0
        return -node.tree_depth

    def put(self, node):
        node.queue_priority = self.generate_priority(node, self._sense)
        return super(BreadthFirstPriorityQueue, self).put(node)

    def put_get(self, node):
        node.queue_priority = self.generate_priority(node, self._sense)
        return super(BreadthFirstPriorityQueue, self).put_get(node)

class DepthFirstPriorityQueue(CustomPriorityQueue):
    """A priority queue implementation that serves nodes in
    depth-first order.

    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    @staticmethod
    def generate_priority(node, sense):
        assert sense in (minimize, maximize)
        assert node.tree_depth >= 0
        return node.tree_depth

    def put(self, node):
        node.queue_priority = self.generate_priority(node, self._sense)
        return super(DepthFirstPriorityQueue, self).put(node)

    def put_get(self, node):
        node.queue_priority = self.generate_priority(node, self._sense)
        return super(DepthFirstPriorityQueue, self).put_get(node)

class FIFOQueue(CustomPriorityQueue):
    """A priority queue implementation that serves nodes in
    first-in, first-out order.

    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    def __init__(self, sense):
        super(FIFOQueue, self).__init__(
            sense,
            _queue_type_=_NoThreadingFIFOQueue)

    @staticmethod
    def generate_priority(node, sense, queue):
        assert sense in (minimize, maximize)
        return -queue._count

    def put(self, node):
        node.queue_priority = self.generate_priority(node,
                                                     self._sense,
                                                     self._queue)
        cnt = super(FIFOQueue, self).put(node)
        assert node.queue_priority == -cnt
        return cnt

    def put_get(self, node):
        node.queue_priority = self.generate_priority(node,
                                                     self._sense,
                                                     self._queue)
        cnt, node_ = super(FIFOQueue, self).put_get(node)
        assert node.queue_priority == -cnt
        return cnt, node_

class LIFOQueue(CustomPriorityQueue):
    """A priority queue implementation that serves nodes in
    last-in, first-out order.

    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    def __init__(self, sense):
        super(LIFOQueue, self).__init__(
            sense,
            _queue_type_=_NoThreadingLIFOQueue)

    @staticmethod
    def generate_priority(node, sense, queue):
        assert sense in (minimize, maximize)
        return queue._count

    def put(self, node):
        node.queue_priority = self.generate_priority(node,
                                                     self._sense,
                                                     self._queue)
        cnt = super(LIFOQueue, self).put(node)
        assert node.queue_priority == cnt
        return cnt

    def put_get(self, node):
        node.queue_priority = self.generate_priority(node,
                                                     self._sense,
                                                     self._queue)
        cnt, node_ = super(LIFOQueue, self).put_get(node)
        assert node.queue_priority == cnt
        return cnt, node_

class RandomPriorityQueue(CustomPriorityQueue):
    """A priority queue implementation that assigns
    a random priority to each incoming node.

    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    @staticmethod
    def generate_priority(node, sense):
        assert sense in (minimize, maximize)
        return random.random()

    def put(self, node):
        node.queue_priority = self.generate_priority(node, self._sense)
        return super(RandomPriorityQueue, self).put(node)

    def put_get(self, node):
        node.queue_priority = self.generate_priority(node, self._sense)
        return super(RandomPriorityQueue, self).put_get(node)

class LocalGapPriorityQueue(CustomPriorityQueue):
    """A priority queue implementation that serves nodes
    with the largest gap between the local objective and
    bound first.

    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    @staticmethod
    def generate_priority(node, sense):
        objective = node.objective
        bound = node.bound
        if sense == minimize:
            gap = objective - bound
        else:
            assert sense == maximize
            gap = bound - objective
        assert not math.isnan(gap)
        return gap

    def put(self, node):
        node.queue_priority = self.generate_priority(node, self._sense)
        return super(LocalGapPriorityQueue, self).put(node)

    def put_get(self, node):
        node.queue_priority = self.generate_priority(node, self._sense)
        return super(LocalGapPriorityQueue, self).put_get(node)
