import array

from pybnb.problem import (new_storage_array,
                           ProblemNode)
from pybnb.mpi_utils import (send_nothing,
                             recv_nothing,
                             recv_data)

try:
    import mpi4py
except ImportError:                               #pragma:nocover
    pass

class ProcessType(object):
    worker = 0
    dispatcher = 1
    @staticmethod
    def is_valid(ptype):
        assert type(ptype) is int
        return ptype in (ProcessType.worker,
                         ProcessType.dispatcher)

class DispatcherAction(object):
    update                    = 21
    solve_finished            = 31
    barrier                   = 41
    finalize                  = 51
    log_info                  = 71
    log_warning               = 72
    log_debug                 = 73
    log_error                 = 74

class WorkerAction(object):
    work             = 110
    nowork           = 210
    receive_solution = 410

class DispatcherProxy(object):

    @staticmethod
    def _init(comm, ptype):
        import mpi4py.MPI
        assert mpi4py.MPI.Is_initialized()
        # make sure there is only one dispatcher
        assert ProcessType.dispatcher == 1
        assert ProcessType.worker == 0
        assert ProcessType.is_valid(ptype)
        assert int(ptype) == ptype
        types_sum = comm.allreduce(ptype, op=mpi4py.MPI.SUM)
        assert types_sum == ProcessType.dispatcher
        dptype, drank = comm.allreduce(sendobj=(ptype, comm.rank),
                                       op=mpi4py.MPI.MAXLOC)
        assert dptype == ProcessType.dispatcher
        if ptype == ProcessType.dispatcher:
            assert drank == comm.rank
        else:
            assert drank != comm.rank

        # tag one worker thread as the "master"
        root_rank = comm.size - 1
        if drank == root_rank:
            root_rank -= 1

        wcomm = comm.Split(0 if comm.rank != drank else 1)

        return drank, root_rank, wcomm

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
        self.CommActionTimer = self._ActionTimer(self)
        self._status = mpi4py.MPI.Status()
        self.comm_time = 0.0
        with self.CommActionTimer:
            (self.dispatcher_rank,
             self.root_worker_comm_rank,
             self.worker_comm) = \
                self._init(comm, ProcessType.worker)

            self.root_worker_worker_comm_rank = None
            if self.comm.rank == self.root_worker_comm_rank:
                self.root_worker_worker_comm_rank = self.worker_comm.rank
            self.root_worker_worker_comm_rank = \
                self.comm.bcast(self.root_worker_worker_comm_rank,
                                root=self.root_worker_comm_rank)

    def update(self, *args, **kwds):
        with self.CommActionTimer:
            return self._update(*args, **kwds)
    def _update(self,
                best_objective,
                previous_bound,
                explored_nodes_count,
                node_states):
        size = 4
        node_states_size = len(node_states)
        if node_states_size > 0:
            for udata in node_states:
                size += 1
                size += len(udata)
        data = new_storage_array(size)
        data[0] = best_objective
        assert float(data[0]) == best_objective
        data[1] = previous_bound
        assert float(data[1]) == previous_bound
        data[2] = explored_nodes_count
        assert data[2] == explored_nodes_count
        assert int(data[2]) == explored_nodes_count
        data[3] = node_states_size
        assert data[3] == node_states_size
        assert int(data[3]) == int(node_states_size)
        if node_states_size > 0:
            pos = 4
            for i in range(node_states_size):
                udata = node_states[i]
                data[pos] = len(udata)
                pos += 1
                data[pos:pos+len(udata)] = udata[:]
                pos += len(udata)
        self.comm.Send([data,mpi4py.MPI.DOUBLE],
                       self.dispatcher_rank,
                       tag=DispatcherAction.update)
        del data
        self.comm.Probe(status=self._status)
        assert not self._status.Get_error()
        tag = self._status.Get_tag()
        if tag == WorkerAction.nowork:
            data = recv_data(self.comm, self._status)
            return float(data[0]), None
        assert tag == WorkerAction.work
        state = recv_data(self.comm, self._status)
        best_objective = ProblemNode._extract_best_objective(state)
        return best_objective, state


    def log_info(self, *args, **kwds):
        with self.CommActionTimer:
            return self._log_info(*args, **kwds)
    def _log_info(self, msg):
        self.comm.Ssend([msg.encode("utf8"),mpi4py.MPI.CHAR],
                        self.dispatcher_rank,
                        tag=DispatcherAction.log_info)

    def log_warning(self, *args, **kwds):
        with self.CommActionTimer:
            return self._log_warning(*args, **kwds)
    def _log_warning(self, msg):
        self.comm.Ssend([msg.encode("utf8"),mpi4py.MPI.CHAR],
                        self.dispatcher_rank,
                        tag=DispatcherAction.log_warning)

    def log_debug(self, *args, **kwds):
        with self.CommActionTimer:
            return self._log_debug(*args, **kwds)
    def _log_debug(self, msg):
        self.comm.Ssend([msg.encode("utf8"),mpi4py.MPI.CHAR],
                        self.dispatcher_rank,
                        tag=DispatcherAction.log_debug)

    def log_error(self, *args, **kwds):
        with self.CommActionTimer:
            return self._log_error(*args, **kwds)
    def _log_error(self, msg):
        self.comm.Ssend([msg.encode("utf8"),mpi4py.MPI.CHAR],
                        self.dispatcher_rank,
                        tag=DispatcherAction.log_error)

    def finalize(self, *args, **kwds):
        with self.CommActionTimer:
            return self._finalize(*args, **kwds)
    def _finalize(self):
        if self.worker_comm.rank == self.root_worker_worker_comm_rank:
            send_nothing(self.comm,
                         self.dispatcher_rank,
                         DispatcherAction.finalize)
        return self.comm.bcast(None, root=self.dispatcher_rank)

    def barrier(self, *args, **kwds):
        with self.CommActionTimer:
            return self._barrier(*args, **kwds)
    def _barrier(self):
        self.worker_comm.Barrier()
        if self.comm.rank == self.root_worker_comm_rank:
            send_nothing(self.comm,
                         self.dispatcher_rank,
                         DispatcherAction.barrier,
                         synchronous=True)
        self.comm.Barrier()

    def solve_finished(self, *args, **kwds):
        with self.CommActionTimer:
            return self._solve_finished(*args, **kwds)
    def _solve_finished(self):
        assert self.worker_comm.rank == self.root_worker_worker_comm_rank
        send_nothing(self.comm,
                     self.dispatcher_rank,
                     DispatcherAction.solve_finished,
                     synchronous=True)
