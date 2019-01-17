"""
A Problem interface for implementing parallel range
reduction on a PyomoProblem during a branch-and-bound solve.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""
import array

from pybnb import (inf, Problem)
from pybnb.node import Node
from pybnb.pyomo.misc import (hash_joblist,
                              add_tmp_component,
                              mpi_partition,
                              create_optimality_bound)
from pybnb.pyomo.problem import PyomoProblem

import pyomo.kernel as pmo
import numpy

try:
    import mpi4py
except ImportError:                               #pragma:nocover
    pass

class RangeReductionProblem(Problem):
    """A specialized implementation of the
    :class:`pybnb.Problem <pybnb.problem.Problem>` interface
    that can be used to perform optimality-based range
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
        tmp_objective_name = add_tmp_component(
            self.problem.pyomo_model,
            "rr_objective",
            tmp_objective)
        # setup optimality bound if necessary
        tmp_optbound_name = None
        tmp_optbound = None
        if self._best_objective != self.infeasible_objective():
            tmp_optbound = create_optimality_bound(
                self,
                self.problem.pyomo_model_objective,
                self._best_objective)
            tmp_optbound_name = add_tmp_component(
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
                                    -inf)
                upper_bounds.append(pmo.value(obj.ub) \
                                    if obj.has_ub() else \
                                    inf)
        lower_bounds = array.array('d', lower_bounds)
        upper_bounds = array.array('d', upper_bounds)

        # verify that everyone has the exact same list
        # (order and values), assumes everything in the list
        # has a well-defined hash
        if self._comm is not None:
            my_joblist_hash = hash_joblist(joblist)
            joblist_hash = self._comm.bcast(my_joblist_hash,
                                            root=0)
            assert joblist_hash == my_joblist_hash
        for i, cid, which in mpi_partition(self._comm,
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
            self._comm.Allreduce([lower_bounds_local,
                                  mpi4py.MPI.DOUBLE],
                                 [lower_bounds,
                                  mpi4py.MPI.DOUBLE],
                                 op=mpi4py.MPI.MAX)
            self._comm.Allreduce([upper_bounds_local,
                                  mpi4py.MPI.DOUBLE],
                                 [upper_bounds,
                                  mpi4py.MPI.DOUBLE],
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
        orig = Node()
        self.save_state(orig)
        node = Node()
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
        node = Node()
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
                                           objective):
        self._best_objective = objective

    def notify_new_best_objective(self,
                                  objective):
        self.notify_new_best_objective_received(objective)

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
        raise NotImplementedError()               #pragma:nocover

    def range_reduction_objective_changed(self, objective):
        """Called to notify that the range reduction routine
        has changed the objective"""
        raise NotImplementedError()               #pragma:nocover

    def range_reduction_constraint_added(self, constraint):
        """Called to notify that the range reduction routine
        has added a constraint"""
        raise NotImplementedError()               #pragma:nocover

    def range_reduction_constraint_removed(self, constraint):
        """Called to notify that the range reduction routine
        has removed a constraint"""
        raise NotImplementedError()               #pragma:nocover

    def range_reduction_get_objects(self):
        """Called to collect the set of objects over which
        to perform range reduction solves"""
        raise NotImplementedError()               #pragma:nocover

    def range_reduction_solve_for_object_bound(self, x):
        """Called to perform a range reduction solve for a
        Pyomo model object"""
        raise NotImplementedError()               #pragma:nocover

    def range_reduction_model_cleanup(self):
        """Called after range reduction has finished to
        allow the user to execute any cleanup to the Pyomo
        model."""
        raise NotImplementedError()               #pragma:nocover

    def range_reduction_process_bounds(self,
                                       objects,
                                       lower_bounds,
                                       upper_bounds):
        """Called to process the bounds obtained by the
        range reduction solves"""
        raise NotImplementedError()               #pragma:nocover
