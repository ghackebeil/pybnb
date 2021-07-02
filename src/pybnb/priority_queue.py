"""
A collection of priority queue implementations that can be
used by the dispatcher.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""
from typing import (
    Type,
    Dict,
    Any,
    Optional,
    Tuple,
    Callable,
    List,
    Iterator,
    Iterable,
    Union,
    TypeVar,
    Generic,
    cast,
)
import random
import collections
import heapq
import math

from sortedcontainers import SortedList
import six

from pybnb.common import minimize, maximize, ProblemSense, inf
from pybnb.node import Node, PriorityType

T = TypeVar("T")


class _NoThreadingMaxPriorityFirstQueue(Generic[T]):
    """A simple priority queue implementation that is not
    thread safe. When the queue is not empty, the item with
    the highest priority is next.

    This queue implementation is not allowed to store None.
    """

    requires_priority = True  # type: bool

    def __init__(self):
        # type: () -> None
        self._count = 0  # type: int
        self._heap = []  # type: List[Tuple[PriorityType, int, T]]

    def _negate(self, priority):
        # type: (PriorityType) -> PriorityType
        if not hasattr(priority, "__iter__"):
            return -priority  # type: ignore
        else:
            return tuple(-v for v in priority)  # type: ignore

    def size(self):
        # type: () -> int
        """Returns the size of the queue."""
        return len(self._heap)

    def put(self, item, priority, _push_=heapq.heappush):
        # type: (T, PriorityType, Any) -> int
        """Puts an item into the queue with the given
        priority. Items placed in the queue may not be
        None. This method returns a unique counter associated
        with each put."""
        if item is None:
            raise ValueError("queue item can not be None")
        cnt = self._count
        self._count += 1
        _push_(self._heap, (self._negate(priority), cnt, item))
        return cnt

    def get(self, _pop_=heapq.heappop):
        # type: (Any) -> Optional[T]
        """Removes and returns the highest priority item in
        the queue, where ties are broken by the order items
        were placed in the queue. If the queue is empty,
        returns None."""
        if len(self._heap) > 0:
            return _pop_(self._heap)[2]
        else:
            return None

    def put_get(self, item, priority, _push_pop_=heapq.heappushpop):
        # type: (T, PriorityType, Any) -> Tuple[int, T]
        """Combines a put and get call, which can be more
        efficient than two separate put and get
        calls. Returns a tuple containing the put and get
        return values."""
        if item is None:
            raise ValueError("queue item can not be None")
        cnt = self._count
        self._count += 1
        if len(self._heap) > 0:
            item_ = _push_pop_(self._heap, (self._negate(priority), cnt, item))[2]
            return cnt, item_
        else:
            return cnt, item

    def next(self):
        # type: () -> Tuple[int, T]
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
        # type: (Callable[[T], bool], bool) -> List[Union[T, Tuple[int, T]]]
        """Removes items from the queue for which
        `func(item)` returns False. The list of items
        removed is returned. If `include_counters` is set to
        True, values in the returned list will have the form
        (cnt, item), where cnt is a unique counter that was
        created for the item when it was added to the
        queue."""
        heap_new = []
        removed = []  # type: List[Union[T, Tuple[int, T]]]
        for priority, cnt, item in self._heap:
            if func(item):
                heap_new.append((priority, cnt, item))
            elif not include_counters:
                removed.append(item)
            else:
                removed.append((cnt, item))
        heapq.heapify(heap_new)
        self._heap = heap_new
        return removed

    def items(self):
        # type: () -> Iterator[T]
        """Iterates over the queued items in arbitrary order
        without modifying the queue."""
        for _, _, item in self._heap:
            yield item


class _NoThreadingFIFOQueue(Generic[T]):
    """A simple first-in, first-out queue implementation
    that is not thread safe.

    This queue implementation is not allowed to store None.
    """

    requires_priority = False  # type: bool

    def __init__(self):
        # type: () -> None
        self._count = 0  # type: int
        self._deque = collections.deque()  # type: collections.deque

    def size(self):
        # type: () -> int
        """Returns the size of the queue."""
        return len(self._deque)

    def put(self, item):
        # type: (T) -> int
        """Puts an item into the queue. Items placed in the
        queue may not be None. This method returns a unique
        counter associated with each put."""
        if item is None:
            raise ValueError("queue item can not be None")
        cnt = self._count
        self._count += 1
        self._deque.append((cnt, item))
        return cnt

    def get(self):
        # type: () -> Optional[T]
        """Removes and returns the next item in the
        queue. If the queue is empty, returns None."""
        if len(self._deque) > 0:
            return self._deque.popleft()[1]
        else:
            return None

    def put_get(self, item):
        # type: (T) -> Tuple[int, T]
        """Combines a put and get call, which can be more
        efficient than two separate put and get
        calls. Returns a tuple containing the put and get
        return values."""
        if item is None:
            raise ValueError("queue item can not be None")
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
        # type: () -> Tuple[int, T]
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
        # type: (Callable[[T], bool], bool) -> List[T]
        """Removes items from the queue for which
        `func(item)` returns False. The list of items
        removed is returned. If `include_counters` is set to
        True, values in the returned list will have the form
        (cnt, item), where cnt is a unique counter that was
        created for the item when it was added to the
        queue."""
        deque_new = collections.deque()  # type: collections.deque
        removed = []
        for cnt, item in self._deque:
            if func(item):
                deque_new.append((cnt, item))
            elif not include_counters:
                removed.append(item)
            else:
                removed.append((cnt, item))
        self._deque = deque_new
        return removed

    def items(self):
        # type: () -> Iterator[T]
        """Iterates over the queued items in arbitrary order
        without modifying the queue."""
        for _, item in self._deque:
            yield item


class _NoThreadingLIFOQueue(Generic[T]):
    """A simple last-in, first-out queue implementation
    that is not thread safe.

    This queue implementation is not allowed to store None.
    """

    requires_priority = False  # type: bool

    def __init__(self):
        # type: () -> None
        self._count = 0  # type: int
        self._items = []  # type: List[Tuple[int, T]]

    def size(self):
        # type: () -> int
        """Returns the size of the queue."""
        return len(self._items)

    def put(self, item):
        # type: (T) -> int
        """Puts an item into the queue. Items placed in the
        queue may not be None. This method returns a unique
        counter associated with each put."""
        if item is None:
            raise ValueError("queue item can not be None")
        cnt = self._count
        self._count += 1
        self._items.append((cnt, item))
        return cnt

    def get(self):
        # type: () -> Optional[T]
        """Removes and returns the next item in the
        queue. If the queue is empty, returns None."""
        if len(self._items) > 0:
            return self._items.pop()[1]
        else:
            return None

    def put_get(self, item):
        # type: (T) -> Tuple[int, T]
        """Combines a put and get call, which can be more
        efficient than two separate put and get
        calls. Returns a tuple containing the put and get
        return values."""
        if item is None:
            raise ValueError("queue item can not be None")
        cnt = self._count
        self._count += 1
        return cnt, item

    def next(self):
        # type: () -> Tuple[int, T]
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
        # type: (Callable[[T], bool], bool) -> List[Union[T, Tuple[int, T]]]
        """Removes items from the queue for which
        `func(item)` returns False. The list of items
        removed is returned. If `include_counters` is set to
        True, values in the returned list will have the form
        (cnt, item), where cnt is a unique counter that was
        created for the item when it was added to the
        queue."""
        items_new = []
        removed = []  # type: List[Union[T, Tuple[int, T]]]
        for cnt, item in self._items:
            if func(item):
                items_new.append((cnt, item))
            elif not include_counters:
                removed.append(item)
            else:
                removed.append((cnt, item))
        self._items = items_new
        return removed

    def items(self):
        # type: () -> Iterator[T]
        """Iterates over the queued items in arbitrary order
        without modifying the queue."""
        for _, item in self._items:
            yield item


SimpleQueueType = Union[
    Type[_NoThreadingMaxPriorityFirstQueue[Node]],
    Type[_NoThreadingFIFOQueue[Node]],
    Type[_NoThreadingLIFOQueue[Node]],
]


class IPriorityQueue(object):
    """The abstract interface for priority queues that store
    node data for the dispatcher."""

    def __init__(self, *args, **kwds):
        raise NotImplementedError  # pragma:nocover

    @staticmethod
    def generate_priority(node, sense, queue):
        # type: (Node, ProblemSense, Any) -> Union[int, float]
        raise NotImplementedError()  # pragma:nocover

    def size(self):
        # type: () -> int
        """Returns the size of the queue."""
        raise NotImplementedError()  # pragma:nocover

    def put(self, node):
        # type: (Node) -> int
        """Puts an node in the queue, possibly updating the
        value of :attr:`queue_priority <pybnb.node.Node.queue_priority>`,
        depending on the queue implementation. This method
        returns a unique counter associated with each
        put."""
        raise NotImplementedError()  # pragma:nocover

    def get(self):
        # type: () -> Optional[Node]
        """Returns the next node in the queue. If the queue
        is empty, returns None."""
        raise NotImplementedError()  # pragma:nocover

    def bound(self):
        # type: () -> Optional[Union[int, float]]
        """Returns the weakest bound of all nodes in the
        queue. If the queue is empty, returns None."""
        raise NotImplementedError()  # pragma:nocover

    def filter(self, func):
        # type: (Callable[[Node], bool]) -> List[Node]
        """Removes nodes from the queue for which
        `func(node)` returns False. The list of nodes
        removed is returned. If the queue is empty or no
        nodes are removed, the returned list will be
        empty."""
        raise NotImplementedError()  # pragma:nocover

    def items(self):
        # type: () -> Iterator[Node]
        """Iterates over the queued nodes in arbitrary order
        without modifying the queue."""
        raise NotImplementedError()  # pragma:nocover


class WorstBoundFirstPriorityQueue(IPriorityQueue):
    """A priority queue implementation that serves nodes
    with the worst bound first.

    Parameters
    ----------
    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    track_bound : bool
        Indicates whether or not to track the global queue
        bound. Note that this particular queue
        implementation always tracks the global bound. This
        argument is ignored.
    """

    def __init__(self, sense, track_bound):
        # type: (ProblemSense, bool) -> None
        assert sense in ProblemSense
        self._sense = sense  # type: ProblemSense
        self._queue = _NoThreadingMaxPriorityFirstQueue[Node]()

    @staticmethod
    def generate_priority(node, sense, queue):
        # type: (Node, ProblemSense, Any) -> Union[int, float]
        bound = node.bound
        assert bound is not None
        assert not math.isnan(bound)
        if sense == minimize:
            return -bound
        else:
            assert sense == maximize
            return bound

    def size(self):
        # type: () -> int
        return self._queue.size()

    def put(self, node):
        # type: (Node) -> int
        node.queue_priority = self.generate_priority(node, self._sense, None)
        return self._queue.put(node, node.queue_priority)

    def get(self):
        # type: () -> Optional[Node]
        return self._queue.get()

    def bound(self):
        # type: () -> Optional[Union[int, float]]
        try:
            return self._queue.next()[1].bound
        except IndexError:
            return None

    def filter(self, func):
        # type: (Callable[[Node], bool]) -> List[Node]
        return cast(List[Node], self._queue.filter(func))

    def items(self):
        # type: () -> Iterator[Node]
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
    track_bound : bool
        Indicates whether or not to track the global queue
        bound.
    """

    def __init__(
        self, sense, track_bound, _queue_type_=_NoThreadingMaxPriorityFirstQueue[Node]
    ):
        # type: (ProblemSense, bool, SimpleQueueType) -> None
        assert sense in ProblemSense
        self._sense = sense
        self._queue = _queue_type_()
        self._sorted_by_bound = None
        if track_bound:
            self._sorted_by_bound = SortedList()

    def size(self):
        # type: () -> int
        return self._queue.size()

    def put(self, node):
        # type: (Node) -> int
        if self._queue.requires_priority:
            priority = node.queue_priority
            if priority is None:
                raise ValueError("A node queue priority is required")
            cnt = self._queue.put(node, priority)  # type: ignore
        else:
            cnt = self._queue.put(node)  # type: ignore
        if self._sorted_by_bound is not None:
            bound = node.bound
            assert bound is not None
            assert not math.isnan(bound)
            if self._sense == maximize:
                self._sorted_by_bound.add((-bound, cnt, node))
            else:
                self._sorted_by_bound.add((bound, cnt, node))
        return cnt

    def get(self):
        # type: () -> Optional[Node]
        if self._queue.size() > 0:
            cnt, tmp_ = self._queue.next()
            node = self._queue.get()
            assert node is not None
            assert tmp_ is node
            if self._sorted_by_bound is not None:
                bound = node.bound
                assert bound is not None
                if self._sense == maximize:
                    self._sorted_by_bound.remove((-bound, cnt, node))
                else:
                    self._sorted_by_bound.remove((bound, cnt, node))
            return node
        else:
            return None

    def bound(self):
        # type: () -> Optional[Union[int, float]]
        if self._sorted_by_bound is not None:
            try:
                return self._sorted_by_bound[0][2].bound
            except IndexError:
                return None
        else:
            if self.size() > 0:
                if self._sense == maximize:
                    return inf
                else:
                    return -inf
            else:
                return None

    def filter(self, func):
        # type: (Callable[[Node], bool]) -> List[Node]
        removed = []
        if self._sorted_by_bound is not None:
            for item in self._queue.filter(func, include_counters=True):
                cnt, node = cast(Tuple[int, Node], item)
                removed.append(node)
                bound = node.bound
                assert bound is not None
                if self._sense == maximize:
                    self._sorted_by_bound.remove((-bound, cnt, node))
                else:
                    self._sorted_by_bound.remove((bound, cnt, node))
        else:
            removed.extend(
                cast(List[Node], self._queue.filter(func, include_counters=False))
            )
        return removed

    def items(self):
        # type: () -> Iterator[Node]
        return self._queue.items()


class BestObjectiveFirstPriorityQueue(CustomPriorityQueue):
    """A priority queue implementation that serves nodes
    with the best objective first.

    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    @staticmethod
    def generate_priority(node, sense, queue):
        # type: (Node, ProblemSense, Any) -> Union[int, float]
        objective = node.objective
        assert objective is not None
        assert not math.isnan(objective)
        if sense == minimize:
            return -objective
        else:
            assert sense == maximize
            return objective

    def put(self, node):
        # type: (Node) -> int
        node.queue_priority = self.generate_priority(node, self._sense, None)
        return super(BestObjectiveFirstPriorityQueue, self).put(node)


class BreadthFirstPriorityQueue(CustomPriorityQueue):
    """A priority queue implementation that serves nodes in
    breadth-first order.

    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    @staticmethod
    def generate_priority(node, sense, queue):
        # type: (Node, ProblemSense, Any) -> Union[int, float]
        tree_depth = node.tree_depth
        assert tree_depth is not None
        assert tree_depth >= 0
        return -tree_depth

    def put(self, node):
        # type: (Node) -> int
        node.queue_priority = self.generate_priority(node, self._sense, None)
        return super(BreadthFirstPriorityQueue, self).put(node)


class DepthFirstPriorityQueue(CustomPriorityQueue):
    """A priority queue implementation that serves nodes in
    depth-first order.

    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    @staticmethod
    def generate_priority(node, sense, queue):
        # type: (Node, ProblemSense, Any) -> Union[int, float]
        tree_depth = node.tree_depth
        assert tree_depth is not None
        assert tree_depth >= 0
        return tree_depth

    def put(self, node):
        # type: (Node) -> int
        node.queue_priority = self.generate_priority(node, self._sense, None)
        return super(DepthFirstPriorityQueue, self).put(node)


class FIFOQueue(CustomPriorityQueue):
    """A priority queue implementation that serves nodes in
    first-in, first-out order.

    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    def __init__(self, sense, track_bound):
        # type: (ProblemSense, bool) -> None
        super(FIFOQueue, self).__init__(
            sense, track_bound, _queue_type_=_NoThreadingFIFOQueue[Node]
        )

    @staticmethod
    def generate_priority(node, sense, queue):
        # type: (Node, ProblemSense, Any) -> Union[int, float]
        return -queue._count

    def put(self, node):
        # type: (Node) -> int
        node.queue_priority = self.generate_priority(node, self._sense, self._queue)
        cnt = super(FIFOQueue, self).put(node)
        assert node.queue_priority == -cnt
        return cnt


class LIFOQueue(CustomPriorityQueue):
    """A priority queue implementation that serves nodes in
    last-in, first-out order.

    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    def __init__(self, sense, track_bound):
        # type: (ProblemSense, bool) -> None
        super(LIFOQueue, self).__init__(
            sense, track_bound, _queue_type_=_NoThreadingLIFOQueue[Node]
        )

    @staticmethod
    def generate_priority(node, sense, queue):
        # type: (Node, ProblemSense, Any) -> Union[int, float]
        return queue._count

    def put(self, node):
        # type: (Node) -> int
        node.queue_priority = self.generate_priority(node, self._sense, self._queue)
        cnt = super(LIFOQueue, self).put(node)
        assert node.queue_priority == cnt
        return cnt


class RandomPriorityQueue(CustomPriorityQueue):
    """A priority queue implementation that assigns
    a random priority to each incoming node.

    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    @staticmethod
    def generate_priority(node, sense, queue):
        # type: (Node, ProblemSense, Any) -> Union[int, float]
        return random.random()

    def put(self, node):
        # type: (Node) -> int
        node.queue_priority = self.generate_priority(node, self._sense, None)
        return super(RandomPriorityQueue, self).put(node)


class LocalGapPriorityQueue(CustomPriorityQueue):
    """A priority queue implementation that serves nodes
    with the largest gap between the local objective and
    bound first.

    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    @staticmethod
    def generate_priority(node, sense, queue):
        # type: (Node, ProblemSense, Any) -> Union[int, float]
        objective = node.objective
        bound = node.bound
        assert objective is not None
        assert bound is not None
        if sense == minimize:
            gap = objective - bound
        else:
            assert sense == maximize
            gap = bound - objective
        assert not math.isnan(gap)
        return gap

    def put(self, node):
        # type: (Node) -> int
        node.queue_priority = self.generate_priority(node, self._sense, None)
        return super(LocalGapPriorityQueue, self).put(node)


class LexicographicPriorityQueue(CustomPriorityQueue):
    """A priority queue implementation that serves nodes
    with the largest gap between the local objective and
    bound first.

    sense : {:obj:`minimize <pybnb.common.minimize>`, :obj:`maximize <pybnb.common.maximize>`}
        The objective sense for the problem.
    """

    def __init__(self, queue_types, sense, track_bound):
        # type: (Iterable[Type[IPriorityQueue]], ProblemSense, bool) -> None
        self._queue_types = tuple(queue_types)  # Tuple[Type[IPriorityQueue], ...]
        assert len(self._queue_types)
        super(LexicographicPriorityQueue, self).__init__(sense, track_bound)

    def _generate_priority(self, node):
        # type: (Node) -> Tuple[Union[int, float], ...]
        return tuple(
            qt.generate_priority(node, self._sense, self._queue)
            for qt in self._queue_types
        )

    def put(self, node):
        # type: (Node) -> int
        node.queue_priority = self._generate_priority(node)
        return super(LexicographicPriorityQueue, self).put(node)


_registered_queue_types = {}  # type: Dict[str, Type[IPriorityQueue]]


def PriorityQueueFactory(name, *args, **kwds):
    # type: (str, Any, Any) -> IPriorityQueue
    """Returns a new instance of the priority queue type
    registered under the given name."""
    if isinstance(name, six.string_types):
        if name not in _registered_queue_types:
            raise ValueError("invalid queue type: %s" % (name))
        return _registered_queue_types[name](*args, **kwds)
    else:
        names = []
        for n_ in name:
            if n_ not in _registered_queue_types:
                raise ValueError("invalid queue type: %s" % (n_))
            if n_ == "custom":
                raise ValueError(
                    "'custom' queue type not "
                    "allowed when defining a "
                    "lexicographic queue strategy"
                )
            names.append(_registered_queue_types[n_])
        if len(names) == 0:
            raise ValueError(
                "Can not define lexicographic queue strategy with empty list"
            )
        return LexicographicPriorityQueue(names, *args, **kwds)


def register_queue_type(name, cls):
    # type: (str, Type[IPriorityQueue]) -> None
    """Registers a new priority queue class with the
    PriorityQueueFactory."""
    if (name in _registered_queue_types) and (_registered_queue_types[name] is not cls):
        raise ValueError(
            "The name '%s' has already been registered"
            "for priority queue type '%s'" % (name, cls)
        )
    _registered_queue_types[name] = cls


register_queue_type("bound", WorstBoundFirstPriorityQueue)
register_queue_type("custom", CustomPriorityQueue)
register_queue_type("objective", BestObjectiveFirstPriorityQueue)
register_queue_type("breadth", BreadthFirstPriorityQueue)
register_queue_type("depth", DepthFirstPriorityQueue)
register_queue_type("fifo", FIFOQueue)
register_queue_type("lifo", LIFOQueue)
register_queue_type("random", RandomPriorityQueue)
register_queue_type("local_gap", LocalGapPriorityQueue)
