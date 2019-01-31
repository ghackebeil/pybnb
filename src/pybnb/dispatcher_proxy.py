"""
A proxy interface to the central dispatcher that is used by
branch-and-bound workers.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""
import array
import collections
import marshal

from pybnb.configuration import config
from pybnb.common import _int_to_termination_condition
from pybnb.node import _SerializedNode
from pybnb.problem import _SolveInfo
from pybnb.mpi_utils import (send_nothing,
                             recv_data)

try:
    import mpi4py
except ImportError:                               #pragma:nocover
    pass

import six

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
        (self.dispatcher_rank,
         self.worker_comm) = self._init(comm, ProcessType.worker)

    def __del__(self):
        if self.worker_comm is not None:
            self.worker_comm.Free()
            self.worker_comm = None

    def update(self,
               best_objective,
               previous_bound,
               solve_info,
               node_list):
        """A proxy to :func:`pybnb.dispatcher.Dispatcher.update`."""
        node_list = [_SerializedNode.to_slots(node_)
                     for node_ in node_list]
        data = marshal.dumps((best_objective,
                              previous_bound,
                              solve_info.data,
                              node_list),
                             config.MARSHAL_PROTOCOL_VERSION)
        self.comm.Send([data,mpi4py.MPI.BYTE],
                       self.dispatcher_rank,
                       tag=DispatcherAction.update)
        self.comm.Probe(status=self._status)
        assert not self._status.Get_error()
        tag = self._status.Get_tag()
        recv_size = self._status.Get_count(mpi4py.MPI.BYTE)
        data = bytearray(recv_size)
        recv_data(self.comm,
                  self._status,
                  datatype=mpi4py.MPI.BYTE,
                  out=data)
        if tag == DispatcherResponse.nowork:
            if six.PY2:
                data_ = str(data)
            else:
                data_ = data
            (best_objective,
             global_bound,
             termination_condition_int,
             solve_info_data) = marshal.loads(data_)
            solve_info = _SolveInfo()
            solve_info.data = array.array('d',solve_info_data)
            return (True,
                    best_objective,
                    (global_bound,
                     _int_to_termination_condition[
                         termination_condition_int],
                     solve_info))
        else:
            assert tag == DispatcherResponse.work
            if six.PY2:
                data_ = str(data)
            else:
                data_ = data
            (best_objective,
             node_slots) = marshal.loads(data_)
            node = _SerializedNode.restore_node(node_slots)
            return False, best_objective, node

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
