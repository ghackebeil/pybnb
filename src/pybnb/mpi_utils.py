"""
Various utility function for MPI.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""

import array

class Message(object):
    """A helper class for probing for / receiving messages"""
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

def recv_nothing(comm, status):
    """A helper function for receiving an empty message
    that matches a probe status.

    Parameters
    ----------
    comm : :class:`mpi4py.MPI.Comm`
        An MPI communicator.
    status : :class:`mpi4py.MPI.Status`
        An MPI status object that has been populated with
        information about the message to be received via a
        probe.
    """
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
    """A helper function for sending an empty message
    with a given tag.

    Parameters
    ----------
    comm : :class:`mpi4py.MPI.Comm`
        An MPI communicator.
    dest : int
        The process rank to send the message to.
    tag : int
        A valid MPI tag to use for the message.
    synchronous : bool, optional
        Indicates whether or not a synchronous MPI send
        should be used. (default=False)
    """
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
    """A helper function for receiving numeric or string
    data sent using the lower-level buffer-based mpi4py
    routines.

    Parameters
    ----------
    comm : :class:`mpi4py.MPI.Comm`
        An MPI communicator.
    status : :class:`mpi4py.MPI.Status`
        An MPI status object that has been populated with
        information about the message to be received via a
        probe.
    datatype : {:obj:`mpi4py.MPI.DOUBLE`, :obj:`mpi4py.MPI.CHAR`}, optional
        An MPI datatype used to interpret the received data. If None,
        :obj:`mpi4py.MPI.DOUBLE` will be used. (default=None)

    Returns
    -------
    :obj:array.array
        The returned array uses typecode "d" when datatype
        is :obj:`mpi4py.MPI.DOUBLE` and uses typecode "B"
        when datatype is :obj:`mpi4py.MPI.CHAR`.
    """
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
