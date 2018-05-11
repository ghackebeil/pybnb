"""
A proxy interface to the central dispatcher that is used by
branch-and-bound workers.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""
import array
import collections

from pybnb.node import Node
from pybnb.mpi_utils import (send_nothing,
                             recv_nothing,
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
        ptype_, dispatcher_rank = comm.allreduce(sendobj=(ptype, comm.rank),
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
            status = recv_nothing(comm)
            return dispatcher_rank, status.Get_source()
        else:
            if worker_comm.rank == 0:
                send_nothing(comm,
                             dispatcher_rank)
            return dispatcher_rank, worker_comm

    class _ActionTimer(object):
        __slots__ = ("_start","_obj")
        def __init__(self, obj):
            self._obj = obj
            self._start = None
        def start(self):
            assert self._start is None
            self._start = mpi4py.MPI.Wtime()
        def stop(self):
            assert self._start is not None
            stop = mpi4py.MPI.Wtime()
            self._obj.comm_time += stop-self._start
            self._start = None
        def __enter__(self):
            self.start()
        def __exit__(self, *args):
            self.stop()

    def __init__(self, comm):
        import mpi4py.MPI
        assert mpi4py.MPI.Is_initialized()
        self.comm = comm
        self.worker_comm = None
        self.CommActionTimer = self._ActionTimer(self)
        self._status = mpi4py.MPI.Status()
        self.comm_time = 0.0
        with self.CommActionTimer:
            (self.dispatcher_rank,
             self.worker_comm) = \
                self._init(comm, ProcessType.worker)

    def __del__(self):
        if self.worker_comm is not None:
            self.worker_comm.Free()
            self.worker_comm = None

    def update(self, *args, **kwds):
        """A proxy to :func:`pybnb.dispatcher.Dispatcher.update`."""
        with self.CommActionTimer:
            return self._update(*args, **kwds)
    def _update(self,
                best_objective,
                previous_bound,
                explored_nodes_count,
                node_data):
        size = 4
        node_data_size = len(node_data)
        if node_data_size > 0:
            for udata in node_data:
                size += 1
                size += len(udata)
        data = numpy.empty(size, dtype=float)
        data[0] = best_objective
        assert float(data[0]) == best_objective
        data[1] = previous_bound
        assert float(data[1]) == previous_bound
        data[2] = explored_nodes_count
        assert data[2] == explored_nodes_count
        assert int(data[2]) == explored_nodes_count
        data[3] = node_data_size
        assert data[3] == node_data_size
        assert int(data[3]) == int(node_data_size)
        if node_data_size > 0:
            pos = 4
            for i in range(node_data_size):
                udata = node_data[i]
                data[pos] = len(udata)
                pos += 1
                data[pos:pos+len(udata)] = udata[:]
                pos += len(udata)
        self.comm.Send([data,mpi4py.MPI.DOUBLE],
                       self.dispatcher_rank,
                       tag=DispatcherAction.update)
        self.comm.Probe(status=self._status)
        assert not self._status.Get_error()
        tag = self._status.Get_tag()
        if tag == DispatcherResponse.nowork:
            data = recv_data(self.comm, self._status)
            return float(data[0]), None
        else:
            assert tag == DispatcherResponse.work
            recv_size = self._status.Get_count(mpi4py.MPI.DOUBLE)
            if len(data) >= recv_size:
                # avoid another allocation and just use a
                # portion of the array was used to send the
                # update data
                data = data[:recv_size]
            else:
                data = numpy.empty(recv_size, dtype=float)
            recv_data(self.comm,
                      self._status,
                      datatype=mpi4py.MPI.DOUBLE,
                      out=data)
            best_objective = Node._extract_best_objective(data)
            return best_objective, data

    def finalize(self, *args, **kwds):
        """A proxy to :func:`pybnb.dispatcher.Dispatcher.finalize`."""
        with self.CommActionTimer:
            return self._finalize(*args, **kwds)
    def _finalize(self):
        if self.worker_comm.rank == 0:
            send_nothing(self.comm,
                         self.dispatcher_rank,
                         DispatcherAction.finalize)
        data = array.array("d",[0])
        self.comm.Bcast([data,mpi4py.MPI.DOUBLE],
                        root=self.dispatcher_rank)
        return float(data[0])

    def log_info(self, *args, **kwds):
        """A proxy to :func:`pybnb.dispatcher.Dispatcher.log_info`."""
        with self.CommActionTimer:
            return self._log_info(*args, **kwds)
    def _log_info(self, msg):
        self.comm.Ssend([msg.encode("utf8"),mpi4py.MPI.CHAR],
                        self.dispatcher_rank,
                        tag=DispatcherAction.log_info)

    def log_warning(self, *args, **kwds):
        """A proxy to :func:`pybnb.dispatcher.Dispatcher.log_warning`."""
        with self.CommActionTimer:
            return self._log_warning(*args, **kwds)
    def _log_warning(self, msg):
        self.comm.Ssend([msg.encode("utf8"),mpi4py.MPI.CHAR],
                        self.dispatcher_rank,
                        tag=DispatcherAction.log_warning)

    def log_debug(self, *args, **kwds):
        """A proxy to :func:`pybnb.dispatcher.Dispatcher.log_debug`."""
        with self.CommActionTimer:
            return self._log_debug(*args, **kwds)
    def _log_debug(self, msg):
        self.comm.Ssend([msg.encode("utf8"),mpi4py.MPI.CHAR],
                        self.dispatcher_rank,
                        tag=DispatcherAction.log_debug)

    def log_error(self, *args, **kwds):
        """A proxy to :func:`pybnb.dispatcher.Dispatcher.log_error`."""
        with self.CommActionTimer:
            return self._log_error(*args, **kwds)
    def _log_error(self, msg):
        self.comm.Ssend([msg.encode("utf8"),mpi4py.MPI.CHAR],
                        self.dispatcher_rank,
                        tag=DispatcherAction.log_error)

    def stop_listen(self, *args, **kwds):
        """Tell the dispatcher to abruptly stop the listen loop."""
        with self.CommActionTimer:
            return self._stop_listen(*args, **kwds)
    def _stop_listen(self):
        assert self.worker_comm.rank == 0
        send_nothing(self.comm,
                     self.dispatcher_rank,
                     DispatcherAction.stop_listen)
