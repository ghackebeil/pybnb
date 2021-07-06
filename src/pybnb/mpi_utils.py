"""
Various utility function for MPI.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""
from typing import Optional, List, Any
import array

from six.moves import xrange as range

# used in various places where we are receiving an empty message,
# initialization is delayed to avoid early mpi4py import
_nothing = None  # type: Optional[List[Any]]


# avoids generating a deprecation warning in python 3.7
def _array_to_string(out):
    """converts an array of bytes to a string"""
    if hasattr(out, "tobytes"):
        # array.tobytes was added in python 3.2
        return out.tobytes().decode("utf8")
    else:
        return out.tostring().decode("utf8")


class Message(object):
    """A helper class for probing for and receiving
    messages. A single instance of this class is meant to be
    reused.

    Parameters
    ----------
    comm : :class:`mpi4py.MPI.Comm`
        The MPI communicator to use.
    """

    __slots__ = ("status", "data", "comm")

    def __init__(self, comm):
        import mpi4py.MPI

        self.comm = comm
        self.status = mpi4py.MPI.Status()
        self.data = None

    def probe(self, **kwds):
        """Perform a blocking test for a message"""
        self.comm.Probe(status=self.status)
        self.data = None

    def recv(self, datatype=None, data=None):
        """Complete the receive for the most recent message
        probe and return the data as a numeric array or a
        string, depending on the datatype keyword.

        Parameters
        ----------
        datatype : {``mpi4py.MPI.DOUBLE``, ``mpi4py.MPI.CHAR``}, optional
            An MPI datatype used to interpret the received
            data. If None, ``mpi4py.MPI.DOUBLE`` will be
            used. (default: None)
        data : array.array or None, optional
            An existing data array to store data into. If
            None, one will be created. (default: None)
        """
        assert not self.status.Get_error()
        if datatype is None:
            count = self.status.Get_count()
        else:
            count = self.status.Get_count(datatype=datatype)
        if count == 0:
            recv_nothing(self.comm, self.status)
        else:
            self.data = recv_data(self.comm, self.status, datatype=datatype, out=data)

    @property
    def tag(self):
        return self.status.Get_tag()

    @property
    def source(self):
        return self.status.Get_source()


def recv_nothing(comm, status):
    """A helper function for receiving an empty
    message. This function is not thread safe.

    Parameters
    ----------
    comm : :class:`mpi4py.MPI.Comm`
        An MPI communicator.
    status : :class:`mpi4py.MPI.Status`
        An MPI status object that has been populated with
        information about the message to be received via a
        probe. If None, a new status object will be created
        and an empty message will be expected from any
        source with any tag. (default: None)

    Returns
    -------
    status : :class:`mpi4py.MPI.Status`
        If the original status argument was not None, it
        will be returned after being updated by the
        receive. Otherwise, the status object that was
        created will be returned.
    """
    global _nothing
    import mpi4py.MPI

    if _nothing is None:
        _nothing = [array.array("B", []), mpi4py.MPI.CHAR]
    assert not status.Get_error()
    assert status.Get_count(mpi4py.MPI.CHAR) == 0
    comm.Recv(_nothing, source=status.Get_source(), tag=status.Get_tag(), status=status)
    assert not status.Get_error()
    assert status.Get_count(mpi4py.MPI.CHAR) == 0
    return status


def send_nothing(comm, dest, tag=0):
    """A helper function for sending an empty message
    with a given tag. This function is not thread safe.

    Parameters
    ----------
    comm : :class:`mpi4py.MPI.Comm`
        An MPI communicator.
    dest : int
        The process rank to send the message to.
    tag : int, optional
        A valid MPI tag to use for the message. (default: 0)
    """
    global _nothing
    import mpi4py.MPI

    if _nothing is None:
        _nothing = [array.array("B", []), mpi4py.MPI.CHAR]
    comm.Send(_nothing, dest, tag=tag)


def recv_data(comm, status, datatype, out=None):
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
    datatype : :class:`mpi4py.MPI.Datatype`
        An MPI datatype used to interpret the received
        data. If the datatype is :obj:`mpi4py.MPI.CHAR`,
        the received data will be converted to a string.
    out : buffer-like object, optional
        A buffer-like object that is compatible with the datatype
        argument and can be passed to comm.Recv. Can only be left
        as None when the datatype is :obj:`mpi4py.MPI.CHAR`.

    Returns
    -------
    string or user-provided data buffer
    """
    import mpi4py.MPI

    assert not status.Get_error()
    size = status.Get_count(datatype)
    if datatype == mpi4py.MPI.CHAR:
        assert out is None
        out = array.array("B", b"\0") * size
    assert (out is not None) and (len(out) >= size)
    comm.Recv(
        [out, datatype], source=status.Get_source(), tag=status.Get_tag(), status=status
    )
    assert not status.Get_error()
    if datatype == mpi4py.MPI.CHAR:
        out = _array_to_string(out)
    return out


def dispatched_partition(comm, items, root=0):
    """A generator that partitions the list of items across
    processes in the communicator. If the communicator size
    is greater than 1, the root process will be yielded no
    items and instead will serve them dynamically by sending
    list indices to workers as work requests are received.

    Parameters
    ----------
    comm : :class:`mpi4py.MPI.Comm` or None
        An MPI communicator or None in the serial
        processing case.
    items : list
        The list of items to partition. This This function
        assumes each process has an identical copy of the
        items list. Therefore, items in the list are not
        transferred (only indices).
    root : integer, optional
        An integer indicating which process rank should be
        designated as the dispatcher. (default: 0)

    Returns
    -------
    string or user-provided data buffer
    """

    assert root >= 0
    N = len(items)
    if N > 0:
        if (comm is None) or (comm.size == 1):
            assert root == 0
            for x in items:
                yield x
        else:
            import mpi4py.MPI

            # it would be pretty easy to refactor this
            # code to avoid this limitation
            assert N <= mpi4py.MPI.COMM_WORLD.Get_attr(mpi4py.MPI.TAG_UB)
            _null = [array.array("b", []), mpi4py.MPI.CHAR]
            last_tag = {}
            if comm.rank == root:
                i = 0
                requests = []
                for dest in range(comm.size):
                    if dest == root:
                        continue
                    last_tag[dest] = i
                    requests.append(comm.Isend(_null, dest, tag=i))
                    i += 1
                status = mpi4py.MPI.Status()
                while i < N:
                    comm.Recv(_null, status=status)
                    last_tag[status.Get_source()] = i
                    requests.append(comm.Isend(_null, status.Get_source(), tag=i))
                    i += 1
                for dest in last_tag:
                    if last_tag[dest] < N:
                        requests.append(comm.Isend(_null, dest, tag=N))
                    requests.append(comm.Irecv(_null, dest))
                mpi4py.MPI.Request.Waitall(requests)
            else:
                status = mpi4py.MPI.Status()
                comm.Recv(_null, source=root, status=status)
                if status.Get_tag() >= N:
                    comm.Send(_null, root)
                else:
                    while status.Get_tag() < N:
                        yield items[status.Get_tag()]
                        comm.Sendrecv(
                            _null, root, recvbuf=_null, source=root, status=status
                        )
