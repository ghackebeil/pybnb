"""
Various tools for implementing branch-and-bound problems
that are based on a pyomo.kernel model.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""

import array
import collections
import hashlib

import pybnb

import pyomo.kernel as pmo

import six
from six.moves import xrange as range

import numpy

try:
    import mpi4py
except ImportError:                               #pragma:nocover
    pass


def _hash_joblist(jobs):
    x = hashlib.sha1()
    for entry in jobs:
        x.update(str(entry).encode())
    return x.hexdigest()

def _add_tmp_component(model, name, obj):
    while hasattr(model, name):
        name = "."+name+"."
    setattr(model, name, obj)
    return name

def _create_optimality_bound(problem,
                             pyomo_objective,
                             best_objective_value):
    optbound = pmo.constraint(body=pyomo_objective)
    if problem.sense() == pybnb.minimize:
        assert pyomo_objective.sense == pmo.minimize
        optbound.ub = best_objective_value
    else:
        assert problem.sense() == pybnb.maximize
        assert pyomo_objective.sense == pmo.maximize
        optbound.lb = best_objective_value
    return optbound

def _mpi_partition(comm, items, root=0):
    assert root >= 0
    N = len(items)
    if N > 0:
        if (comm is None) or \
           (comm.size == 1):
            assert root == 0
            for x in items:
                yield x
        else:
            import mpi4py.MPI
            _null = [array.array('b',[]),mpi4py.MPI.CHAR]
            last_tag = {}
            if comm.rank == root:
                i = 0
                for dest in range(1, comm.size):
                    last_tag[dest] = i
                    comm.Send(_null, dest, tag=i)
                    i += 1
                status = mpi4py.MPI.Status()
                while i < N:
                    comm.Recv(_null, status=status)
                    last_tag[status.Get_source()] = i
                    comm.Send(_null, status.Get_source(), tag=i)
                    i += 1
                for dest in last_tag:
                    if last_tag[dest] < N:
                        comm.Send(_null, dest, tag=N)
            else:
                status = mpi4py.MPI.Status()
                comm.Recv(_null, source=0, status=status)
                while status.Get_tag() < N:
                    yield items[status.Get_tag()]
                    comm.Sendrecv(_null, 0, recvbuf=_null, source=0, status=status)

def generate_cids(model,
                  prefix=(),
                  **kwds):
    """Generate forward and reverse mappings between model
    components and deterministic, unique identifiers that
    are safe to serialize or use as dictionary keys."""
    object_to_cid = pmo.ComponentMap()
    cid_to_object = collections.OrderedDict()
    traversal = model.preorder_traversal(return_key=True, **kwds)
    obj_ = six.next(traversal)[1]
    assert obj_ is model
    object_to_cid[model] = prefix
    cid_to_object[prefix] = model
    for key, obj in traversal:
        parent = obj.parent
        cid_ = object_to_cid[obj] = object_to_cid[parent]+(key,)
        cid_to_object[cid_] = obj
    return object_to_cid, cid_to_object

class PyomoProblem(pybnb.Problem):
    """An extension of the :class:`pybnb.Problem
    <pybnb.problem.Problem>` base class for defining
    problems with a core Pyomo model."""

    def __init__(self, *args, **kwds):
        super(PyomoProblem, self).__init__(*args, **kwds)
        self.__pyomo_object_to_cid = None
        self.__cid_to_pyomo_object = None
        self.update_pyomo_object_cids()

    def update_pyomo_object_cids(self):
        (self.__pyomo_object_to_cid,
         self.__cid_to_pyomo_object) = \
            generate_cids(self.pyomo_model,
                          active=None)
        assert len(set(self.pyomo_object_to_cid.values())) == \
            len(self.__pyomo_object_to_cid)
        assert len(self.cid_to_pyomo_object) == \
            len(self.__pyomo_object_to_cid)

    @property
    def pyomo_object_to_cid(self):
        """The map from pyomo model object to component id."""
        return self.__pyomo_object_to_cid

    @property
    def cid_to_pyomo_object(self):
        """The map from component id to pyomo model object."""
        return self.__cid_to_pyomo_object

    #
    # Abstract Methods
    #

    @property
    def pyomo_model(self):
        """Returns the pyomo model for this problem."""
        raise NotImplementedError()                    #pragma:nocover

    @property
    def pyomo_model_objective(self):
        """Returns the pyomo model objective for this
        problem."""
        raise NotImplementedError()                    #pragma:nocover

class RangeReductionProblem(pybnb.Problem):
    """A specialized implementation of the
    :class:`pybnb.Problem <pybnb.problem.Problem>` base
    class that can be used to perform optimality-based range
    reduction on a fully implemented :class:`PyomoProblem`
    by defining additional abstract methods."""

    def __init__(self,
                 problem,
                 best_objective,
                 comm=None):
        assert isinstance(problem, PyomoProblem)
        self.problem = problem
        assert best_objective != self.unbounded_objective
        self._best_objective = float(best_objective)
        self._comm = comm
        if self._comm is not None:
            import mpi4py.MPI
        self._current_node = None

    def _notify_continue_listen(self, node):
        assert (self._comm is not None) and \
            (self._comm.size > 1)
        data = numpy.array([True,
                            self._best_objective,
                            len(node.state)],
                           dtype=float)
        assert data[0] == True
        assert data[1] == self._best_objective
        assert data[2] == len(node.state)
        self._comm.Bcast([data, mpi4py.MPI.DOUBLE],
                         root=self._comm.rank)
        self._comm.Bcast([node.state, mpi4py.MPI.DOUBLE],
                         root=self._comm.rank)

    def _notify_stop_listen(self):
        assert (self._comm is not None) and \
            (self._comm.size > 1)
        data = numpy.array([False,
                            self._best_objective,
                            0],
                           dtype=float)
        assert data[0] == False
        assert data[1] == self._best_objective
        assert data[2] == 0
        self._comm.Bcast([data, mpi4py.MPI.DOUBLE],
                         root=self._comm.rank)

    def _tighten_bounds(self):
        self.range_reduction_model_setup()
        assert self._best_objective != self.unbounded_objective()
        # setup objective
        assert self.problem.pyomo_model_objective.active
        self.problem.pyomo_model_objective.deactivate()
        tmp_objective = pmo.objective()
        tmp_objective_name = _add_tmp_component(
            self.problem.pyomo_model,
            "rr_objective",
            tmp_objective)
        # setup optimality bound if necessary
        tmp_optbound_name = None
        tmp_optbound = None
        if self._best_objective != self.infeasible_objective():
            tmp_optbound = _create_optimality_bound(
                self,
                self.problem.pyomo_model_objective,
                self._best_objective)
            tmp_optbound_name = _add_tmp_component(
                self.problem.pyomo_model,
                "optimality_bound",
                tmp_optbound)
            self.range_reduction_constraint_added(tmp_optbound)
        try:
            return self._tighten_bounds_impl(tmp_objective)
        finally:
            # reset objective
            delattr(self.problem.pyomo_model, tmp_objective_name)
            self.problem.pyomo_model_objective.activate()
            self.range_reduction_objective_changed(
                self.problem.pyomo_model_objective)
            # remove optimality bound if it was added
            if tmp_optbound is not None:
                self.range_reduction_constraint_removed(tmp_optbound)
                delattr(self.problem.pyomo_model, tmp_optbound_name)
            self.range_reduction_model_cleanup()

    def _tighten_bounds_impl(self, tmp_objective):
        objlist = self.range_reduction_get_objects()
        joblist = []
        objects = []
        lower_bounds = []
        upper_bounds = []
        objects_seen = set()
        for i, val in enumerate(objlist):
            obj = None
            include = False
            val = val if type(val) is tuple else (val,True,True)
            assert len(val) == 3
            obj = val[0]
            cid = self.problem.pyomo_object_to_cid[obj]
            if val[1]:
                include = True
                joblist.append((i,cid,'L'))
            if val[2]:
                include = True
                joblist.append((i,cid,'U'))
                joblist.append((i,cid,'U'))
            if include:
                assert obj is not None
                assert id(obj) not in objects_seen
                objects_seen.add(id(obj))
                objects.append(obj)
                lower_bounds.append(pmo.value(obj.lb) \
                                    if obj.has_lb() else \
                                    -pybnb.inf)
                upper_bounds.append(pmo.value(obj.ub) \
                                    if obj.has_ub() else \
                                    pybnb.inf)
        lower_bounds = array.array('d', lower_bounds)
        upper_bounds = array.array('d', upper_bounds)

        # verify that everyone has the exact same list
        # (order and values), assumes everything in the list
        # has a well-defined hash
        if self._comm is not None:
            my_joblist_hash = _hash_joblist(joblist)
            joblist_hash = self._comm.bcast(my_joblist_hash,
                                            root=0)
            assert joblist_hash == my_joblist_hash
        for i, cid, which in _mpi_partition(self._comm,
                                            joblist):
            obj = self.problem.cid_to_pyomo_object[cid]
            tmp_objective.expr = obj
            if which == 'L':
                tmp_objective.sense = pmo.minimize
            else:
                assert which == 'U'
                tmp_objective.sense = pmo.maximize
            self.range_reduction_objective_changed(tmp_objective)
            bound = self.range_reduction_solve_for_object_bound(obj)
            if bound is not None:
                if which == 'L':
                    lower_bounds[i] = bound
                else:
                    assert which == 'U'
                    upper_bounds[i] = bound
        if self._comm is not None:
            lower_bounds_local = lower_bounds
            upper_bounds_local = upper_bounds
            lower_bounds = array.array('d', lower_bounds)
            upper_bounds = array.array('d', upper_bounds)
            self._comm.Allreduce([lower_bounds_local, mpi4py.MPI.DOUBLE],
                                 [lower_bounds, mpi4py.MPI.DOUBLE],
                                 op=mpi4py.MPI.MAX)
            self._comm.Allreduce([upper_bounds_local, mpi4py.MPI.DOUBLE],
                                 [upper_bounds, mpi4py.MPI.DOUBLE],
                                 op=mpi4py.MPI.MIN)

        return objects, lower_bounds, upper_bounds

    #
    # Interface
    #

    def listen(self, root=0):
        """Listen for requests to run range reduction. All
        processes within the communicator, except for the
        root process, should call this method.

        Parameters
        ----------
        root : int
            The rank of the process acting as the
            root. The root process should not call this
            function.
        """
        assert self._comm.size > 1
        assert self._comm.rank != root
        orig = pybnb.node.Node()
        self.save_state(orig)
        node = pybnb.node.Node()
        try:
            data = numpy.empty(3,dtype=float)
            self._comm.Bcast([data,mpi4py.MPI.DOUBLE],
                             root=root)
            again = bool(data[0])
            self._best_objective = float(data[1])
            node_size = int(data[2])
            while again:
                node.resize(node_size)
                self._comm.Bcast([node.state,mpi4py.MPI.DOUBLE],
                                 root=root)
                self.load_state(node)
                self._tighten_bounds()
                self._comm.Bcast([data,mpi4py.MPI.DOUBLE],
                                 root=root)
                again = bool(data[0])
                self._best_objective = float(data[1])
                node_size = int(data[2])
        finally:
            self.load_state(orig)

    #
    # Implement Problem abstract methods
    #

    def sense(self):
        return self.problem.sense()

    def objective(self):
        return self._best_objective

    def bound(self):
        # tell the listeners to start bounds tightening
        node = pybnb.node.Node()
        self.save_state(node)
        continue_loop = True
        while continue_loop:
            if (self._comm is not None) and \
               (self._comm.size > 1):
                self._notify_continue_listen(node)
            continue_loop = self.range_reduction_process_bounds(
                *self._tighten_bounds())
            self.save_state(node)
        return self.problem.bound()

    def save_state(self, node):
        self.problem.save_state(node)

    def load_state(self, node):
        self.problem.load_state(node)

    def branch(self, parent):
        return self.problem.branch(parent)

    def notify_new_best_objective_received(self,
                                           worker_comm,
                                           best_objective):
        self._best_objective = best_objective

    def notify_new_best_objective(self,
                                  worker_comm,
                                  best_objective):
        self.notify_new_best_objective_received(worker_comm,
                                                best_objective)

    def notify_solve_finished(self,
                              comm,
                              worker_comm,
                              results):
        if (self._comm is not None) and \
           (self._comm.size > 1):
            self._notify_stop_listen()

    #
    # Abstract Methods
    #

    def range_reduction_model_setup(self):
        """Called prior to starting range reduction solves
        to set up the Pyomo model"""
        raise NotImplementedError()                            #pragma:nocover

    def range_reduction_objective_changed(self, objective):
        """Called to notify that the range reduction routine
        has changed the objective"""
        raise NotImplementedError()                            #pragma:nocover

    def range_reduction_constraint_added(self, constraint):
        """Called to notify that the range reduction routine
        has added a constraint"""
        raise NotImplementedError()                            #pragma:nocover

    def range_reduction_constraint_removed(self, constraint):
        """Called to notify that the range reduction routine
        has removed a constraint"""
        raise NotImplementedError()                            #pragma:nocover

    def range_reduction_get_objects(self):
        """Called to collect the set of objects over which
        to perform range reduction solves"""
        raise NotImplementedError()                            #pragma:nocover

    def range_reduction_solve_for_object_bound(self, x):
        """Called to perform a range reduction solve for a
        Pyomo model object"""
        raise NotImplementedError()                            #pragma:nocover

    def range_reduction_model_cleanup(self):
        """Called after range reduction has finished to
        allow the user to execute any cleanup to the Pyomo
        model."""
        raise NotImplementedError()                            #pragma:nocover

    def range_reduction_process_bounds(self,
                                       objects,
                                       lower_bounds,
                                       upper_bounds):
        """Called to process the bounds obtained by the
        range reduction solves"""
        raise NotImplementedError()                            #pragma:nocover
