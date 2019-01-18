"""
A proxy interface to the central dispatcher that is used by
branch-and-bound workers.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""
import collections

from pybnb.common import _int_to_termination_condition
from pybnb.node import Node
from pybnb.problem import _SolveInfo
from pybnb.mpi_utils import (send_nothing,
                             recv_data)

import numpy

try:
    import mpi4py
except ImportError:                               #pragma:nocover
    pass

_ProcessType = collections.namedtuple(
    "_ProcessType",
    ["worker",
     "dispatcher"])
ProcessType = _ProcessType(
    worker     = 0,
    dispatcher = 1)
"""A namespace of typecodes that are used to categorize
processes during dispatcher startup."""

_DispatcherAction = collections.namedtuple(
    "_DispatcherAction",
    ["update",
     "finalize",
     "log_info",
     "log_warning",
     "log_debug",
     "log_error",
     "stop_listen"])

DispatcherAction = _DispatcherAction(
    update                    = 111,
    finalize                  = 211,
    log_info                  = 311,
    log_warning               = 411,
    log_debug                 = 511,
    log_error                 = 611,
    stop_listen               = 711)
"""A namespace of typecodes that are used to categorize
messages received by the dispatcher from workers."""

_DispatcherResponse = collections.namedtuple(
    "_DispatcherResponse",
    ["work",
     "nowork"])
DispatcherResponse = _DispatcherResponse(
    work             = 1111,
    nowork           = 2111)
"""A namespace of typecodes that are used to categorize
responses received by workers from the dispatcher."""

class DispatcherProxy(object):
    """A proxy class for interacting with the central
    dispatcher via message passing."""

    @staticmethod
    def _init(comm, ptype):
        """Broadcasts the dispatcher rank to everyone and
        sets up a worker communicator that excludes the
        single dispatcher."""
        import mpi4py.MPI
        assert mpi4py.MPI.Is_initialized()
        assert len(ProcessType) == 2
        assert ProcessType.dispatcher == 1
        assert ProcessType.worker == 0
        assert ptype in ProcessType
        ptype_, dispatcher_rank = comm.allreduce(
            sendobj=(ptype, comm.rank),
            op=mpi4py.MPI.MAXLOC)
        assert ptype_ == ProcessType.dispatcher
        color = None
        if ptype == ProcessType.dispatcher:
            assert dispatcher_rank == comm.rank
            color = 1
        else:
            assert dispatcher_rank != comm.rank
            color = 0
        assert color is not None
        worker_comm = comm.Split(color)
        if color == 1:
            worker_comm.Free()
            return dispatcher_rank
        else:
            return dispatcher_rank, worker_comm

    def __init__(self, comm):
        import mpi4py.MPI
        assert mpi4py.MPI.Is_initialized()
        self.comm = comm
        self.worker_comm = None
        self._status = mpi4py.MPI.Status()
        self._update_buffer = None
        (self.dispatcher_rank,
         self.worker_comm) = self._init(comm, ProcessType.worker)

    def __del__(self):
        if self.worker_comm is not None:
            self.worker_comm.Free()
            self.worker_comm = None
        self.clear_cache()

    def clear_cache(self):
        self._update_buffer = None

    def update(self,
               best_objective,
               previous_bound,
               solve_info,
               node_data_list):
        """A proxy to :func:`pybnb.dispatcher.Dispatcher.update`."""
        size = 3 + _SolveInfo._data_size
        node_count = len(node_data_list)
        if node_count > 0:
            for node_data_ in node_data_list:
                size += 1
                size += len(node_data_)
        if (self._update_buffer is None) or \
           len(self._update_buffer) < size:
            self._update_buffer = numpy.empty(size, dtype=float)
        data = self._update_buffer
        data[0] = best_objective
        assert float(data[0]) == best_objective
        data[1] = previous_bound
        assert float(data[1]) == previous_bound
        data[2] = node_count
        assert data[2] == node_count
        assert int(data[2]) == int(node_count)
        data[3:(_SolveInfo._data_size)+3] = solve_info.data
        if node_count > 0:
            pos = _SolveInfo._data_size+3
            for node_data in node_data_list:
                data[pos] = len(node_data)
                pos += 1
                data[pos:pos+len(node_data)] = node_data
                pos += len(node_data)

        self.comm.Send([data,mpi4py.MPI.DOUBLE],
                       self.dispatcher_rank,
                       tag=DispatcherAction.update)
        self.comm.Probe(status=self._status)
        assert not self._status.Get_error()
        tag = self._status.Get_tag()
        if tag == DispatcherResponse.nowork:
            data = recv_data(self.comm, self._status)
            best_objective = float(data[0])
            global_bound = float(data[1])
            termination_condition = \
                _int_to_termination_condition[int(data[2])]
            solve_info = _SolveInfo()
            solve_info.data[:] = data[3:]
            return (True,
                    best_objective,
                    (global_bound,
                     termination_condition,
                     solve_info))
        else:
            assert tag == DispatcherResponse.work
            recv_size = self._status.Get_count(mpi4py.MPI.DOUBLE)
            if len(self._update_buffer) < recv_size:
                self._update_buffer = numpy.empty(recv_size, dtype=float)
            # Note that this function returns a node data
            # array that is a view on its own update
            # buffer. Thus, it assumes that the caller is no
            # longer using the node data view that was
            # returned when the next update is called
            # (because it will be corrupted).
            data = self._update_buffer[:recv_size]
            recv_data(self.comm,
                      self._status,
                      datatype=mpi4py.MPI.DOUBLE,
                      out=data)
            best_objective = Node._extract_best_objective(data)
            return False, best_objective, data

    def log_info(self, msg):
        """A proxy to :func:`pybnb.dispatcher.Dispatcher.log_info`."""
        self.comm.Ssend([msg.encode("utf8"),mpi4py.MPI.CHAR],
                        self.dispatcher_rank,
                        tag=DispatcherAction.log_info)

    def log_warning(self, msg):
        """A proxy to :func:`pybnb.dispatcher.Dispatcher.log_warning`."""
        self.comm.Ssend([msg.encode("utf8"),mpi4py.MPI.CHAR],
                        self.dispatcher_rank,
                        tag=DispatcherAction.log_warning)

    def log_debug(self, msg):
        """A proxy to :func:`pybnb.dispatcher.Dispatcher.log_debug`."""
        self.comm.Ssend([msg.encode("utf8"),mpi4py.MPI.CHAR],
                        self.dispatcher_rank,
                        tag=DispatcherAction.log_debug)

    def log_error(self, msg):
        """A proxy to :func:`pybnb.dispatcher.Dispatcher.log_error`."""
        self.comm.Ssend([msg.encode("utf8"),mpi4py.MPI.CHAR],
                        self.dispatcher_rank,
                        tag=DispatcherAction.log_error)

    def stop_listen(self):
        """Tell the dispatcher to abruptly stop the listen loop."""
        assert self.worker_comm.rank == 0
        send_nothing(self.comm,
                     self.dispatcher_rank,
                     tag=DispatcherAction.stop_listen)
