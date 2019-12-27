try:
    import mpi4py  # noqa: F401

    mpi_available = True
except ImportError:  # pragma:nocover
    mpi_available = False
