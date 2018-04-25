try:
    import mpi4py
    mpi_available = True
except ImportError:                               #pragma:nocover
    mpi_available = False
