import itertools
import array

import six
from six import next
from six.moves import xrange as range

class Message(object):
    __slots__ = ("status","data","comm")
    def __init__(self, comm):
        import mpi4py.MPI
        self.comm = comm
        self.status = mpi4py.MPI.Status()
        self.data = None
    def probe(self):
        self.comm.Probe(status=self.status)
        self.data = None
    def recv(self, datatype=None):
        assert not self.status.Get_error()
        if datatype is None:
            count = self.status.Get_count()
        else:
            count = self.status.Get_count(datatype=datatype)
        if count == 0:
            recv_nothing(self.comm,
                         self.status)
        else:
            self.data = recv_data(self.comm,
                                  self.status,
                                  datatype=datatype)
    @property
    def tag(self): return self.status.Get_tag()
    @property
    def source(self): return self.status.Get_source()

def partition(comm, items, root=0):
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

def recv_nothing(comm, status):
    import mpi4py.MPI
    if recv_nothing._nothing is None:
        recv_nothing._nothing = [array.array("B",[]),
                                 mpi4py.MPI.CHAR]
    assert not status.Get_error()
    assert status.Get_count(mpi4py.MPI.CHAR) == 0
    comm.Recv(recv_nothing._nothing,
              source=status.Get_source(),
              tag=status.Get_tag(),
              status=status)
    assert not status.Get_error()
    assert status.Get_count(mpi4py.MPI.CHAR) == 0
recv_nothing._nothing = None

def send_nothing(comm, dest, tag, synchronous=False):
    import mpi4py.MPI
    if send_nothing._nothing is None:
        send_nothing._nothing = [array.array("B",[]),
                                 mpi4py.MPI.CHAR]
    if not synchronous:
        Send = comm.Send
    else:
        Send = comm.Ssend
    Send(send_nothing._nothing,
         dest,
         tag=tag)
send_nothing._nothing = None

def recv_data(comm, status, datatype=None):
    import mpi4py.MPI
    assert not status.Get_error()
    if datatype is None:
        datatype = mpi4py.MPI.DOUBLE
    size = status.Get_count(datatype)
    if datatype == mpi4py.MPI.DOUBLE:
        data = array.array("d",[0])*size
    else:
        assert datatype == mpi4py.MPI.CHAR
        data = array.array("B",b"\0")*size
    comm.Recv([data,datatype],
              source=status.Get_source(),
              tag=status.Get_tag(),
              status=status)
    assert not status.Get_error()
    if datatype == mpi4py.MPI.CHAR:
        data = data.tostring().decode("utf8")
    return data
