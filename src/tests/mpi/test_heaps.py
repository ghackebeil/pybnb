import itertools

import pytest
from runtests.mpi import MPITest

import six
from six.moves import zip

import pybnb

from .common import mpi_available

def left_child(i):
    return 2*i + 1

def right_child(i):
    return 2*i + 2

def log2floor(n):
    assert n > 0
    return n.bit_length() - 1

def height(size):
    return log2floor(size)

def set_none(heap, i):
    if i < len(heap):
        heap[i] = None
        set_none(heap, left_child(i))
        set_none(heap, right_child(i))

def set_one(heap, i):
    if i < len(heap):
        if heap[i] is not None:
            heap[i] = 1
            set_one(heap, left_child(i))
            set_one(heap, right_child(i))

def is_terminal(heap, i):
    N = len(heap)
    c1 = left_child(i)
    c2 = right_child(i)
    return ((c1 >= N) or (heap[c1] is None)) and \
           ((c2 >= N) or (heap[c2] is None))

def get_bound(heap, how=min):
    N = len(heap)
    assert N >= 1
    assert heap[0] is not None
    terminal_bounds = []
    for i in range(N):
        if (heap[i] is not None) and is_terminal(heap, i):
            terminal_bounds.append(heap[i])
    return how(terminal_bounds)

def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def gen_heaps(k):
    for heap_size in range(1,2**(k+1)):
        h = height(heap_size)
        for level_none in range(h,h+1):
            level_nodes = list(filter(lambda x: x < heap_size,
                                      range((2**level_none)-1,(2**(level_none+1))-1)))
            for none_list in sorted(powerset(level_nodes)):
                if len(none_list) == len(level_nodes):
                    continue
                heap_master = [0]*heap_size
                for i in none_list:
                    set_none(heap_master, i)
                for level in range(0,h+1):
                    nodes = filter(lambda x: (x < heap_size) and (heap_master[x] is not None),
                                   range((2**level)-1,(2**(level+1))-1))
                    for nodes_list in sorted(powerset(nodes)):
                        if (len(nodes_list) == 0) and \
                           level != 0:
                            continue
                        heap = [0]*heap_size
                        for i in none_list:
                            set_none(heap, i)
                        for i in nodes_list:
                            set_one(heap, i)
                        yield heap

class DiscreteMin(pybnb.Problem):

    def __init__(self,
                 objectives,
                 bound_bheap,
                 default_objective):
        assert len(bound_bheap) >= 1
        self._objectives = objectives
        self._bound_bheap = bound_bheap
        self._default_objective = default_objective
        self._heap_idx = 0

    #
    # Implement Problem abstract methods
    #

    def sense(self):
        return pybnb.minimize

    def objective(self):
        return self._objectives.get(self._heap_idx,
                                    self._default_objective)

    def bound(self):
        return self._bound_bheap[self._heap_idx]

    def save_state(self, node):
        node.resize(1)
        node.state[0] = self._heap_idx

    def load_state(self, node):
        assert len(node.state) == 1
        self._heap_idx = int(node.state[0])

    def branch(self, parent):
        i = self._heap_idx
        assert i >= 0
        assert i < len(self._bound_bheap)
        left_idx =  2*i + 1
        children = []
        if (left_idx < len(self._bound_bheap)) and \
           (self._bound_bheap[left_idx] is not None):
            child = parent.new_child(size=1)
            child.state[0] = left_idx
            children.append(child)
        right_idx = 2*i + 2
        if (right_idx < len(self._bound_bheap)) and \
           (self._bound_bheap[right_idx] is not None):
            child = parent.new_child(size=1)
            child.state[0] = right_idx
            children.append(child)
        return children

def _test_heaps(comm):
    solver = pybnb.Solver(comm=comm)
    if comm.rank == 0:
        pass
    elif comm.rank == 1:
        pass
    elif comm.rank == 3:
        pass
    for heap in gen_heaps(2):
        heap_bound = get_bound(heap)
        node_list = [None, len(heap)] + [i for i in range(len(heap))
                                         if heap[i] is not None]
        for default_objective in [None, 2]:
            for objective_node in node_list:
                if objective_node is not None:
                    problem = DiscreteMin({objective_node: 1},
                                          heap,
                                          default_objective=2)
                else:
                    problem = DiscreteMin({},
                                          heap,
                                          default_objective=1)
                results = solver.solve(problem, log=None)
                if objective_node == len(heap):
                    assert results.objective == 2
                else:
                    assert results.objective == 1
                assert results.bound == heap_bound

if mpi_available:
    @MPITest(commsize=[1, 2, 4])
    def test_heaps_comm(comm):
        _test_heaps(comm)
