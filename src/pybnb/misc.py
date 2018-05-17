"""
Miscellaneous utilities used for development.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""

import logging

def metric_fmt(num, unit="s", digits=1):
    """Format and scale output with metric prefixes.

    Example
    -------

    >>> metric_fmt(0)
    '0.0 s'
    >>> metric_fmt(0.002, unit='B')
    '2.0 mB'
    >>> metric_fmt(2001, unit='B')
    '2.0 KB'
    >>> metric_fmt(2001, unit='B', digits=3)
    '2.001 KB'

    """
    if num is None:
        return "<unknown>"
    prefix = ""
    if (num >= 1.0) or (num == 0.0):
        if num >= 1000.0:
            num /= 1000.0
            for prefix in ['K','M','G','T','P','E','Z','Yi']:
                if abs(num) < 1000.0:
                    break
                num /= 1000.0
    else:
        num *= 1000.0
        for prefix in ['m','u','n','p','f']:
            if abs(num) > 1:
                break
            num *= 1000.0
    return ("%."+str(digits)+"f %s%s") % (num, prefix, unit)

def get_gap_labels(gap,
                   key="gap",
                   format="f"):
    """Get format strings with enough size and precision to print
    a given gap tolerance."""
    gap_length = 10
    gap_digits = 0
    while gap < (10**(-gap_digits+1)):
        gap_digits += 1
        if gap_length - gap_digits < 5:
            gap_length += 1
    gap_label_str = "{"+key+":>"+str(gap_length)+"}"
    gap_number_str = "{"+key+":>"+str(gap_length)+"." + \
                     str(gap_digits)+format+"}"
    return gap_label_str, gap_number_str

class _NullCM(object):
    """A context manager that does nothing"""
    def __init__(self, obj):
        self.obj = obj
    def __enter__(self):
        return self.obj
    def __exit__(self, *args):
        pass

def as_stream(stream, **kwds):
    """A utility for handling function arguments that can be
    a filename or a file object. This function is meant to be
    used in the context of a with statement.

    Parameters
    ----------
    stream : file-like object or string
        An existing file-like object or the name of a file
        to open.
    **kwds
        Additional keywords passed to the built-in function
        ``open`` when the `stream` keyword is a filename.

    Returns
    -------
    file-like object
        A file-like object that can be written to. If the
        input argument was originally an open file, a dummy
        context will wrap the file object so that it will
        not be closed upon exit of the with block.

    Example
    -------

    >>> import tempfile
    >>> with tempfile.NamedTemporaryFile() as f:
    ...     # pass a file
    ...     with as_stream(f) as g:
    ...         assert g is f
    ...     assert not f.closed
    ...     f.close()
    ...     # pass a filename
    ...     with as_stream(f.name) as g:
    ...         assert not g.closed
    ...     assert g.closed

    """
    import six
    if isinstance(stream, six.string_types):
        return open(stream,"w")
    else:
        return _NullCM(stream)

def get_default_args(func):
    """Get the default arguments for a function as a
    dictionary mapping argument name to default value.

    Example
    -------

    >>> def f(a, b=None):
    ...     pass
    >>> get_default_args(f)
    {'b': None}

    """
    import inspect
    import six
    if six.PY3:
        signature = inspect.signature(func)
        return {k: v.default
                for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty}
    else:
        a = inspect.getargspec(func)
        if a.defaults is None:
            return {}
        return dict(zip(a.args[-len(a.defaults):],a.defaults))

def get_keyword_docs(doc):
    """Parses a numpy-style docstring to summarize
    information in the 'Parameters' section into a dictionary."""
    import re
    import yaml
    lines = doc.splitlines()
    for i_start, line in enumerate(lines):
        if (line.strip() == "Parameters") and \
           (lines[i_start+1].strip() == "----------"):
            i_start = i_start + 2
            break
    else:                                              #pragma:nocover
        assert False
    for i_stop, line in enumerate(lines[i_start:],i_start):
        if i_stop <= i_start+1:
            continue
        if line.strip() == "":
            break
    else:                                              #pragma:nocover
        assert False
    args = {}
    assert re.match(r".+ : .+(, optional)?", lines[i_start])
    last = lines[i_start].split(' : ')[0].strip()
    args[last] = ""
    i = i_start + 1
    while i != i_stop:
        if re.match(r".+ : .+(, optional)?", lines[i]):
            args[last] = args[last].strip()
            last = lines[i].split(' : ')[0].strip()
            args[last] = ""
        else:
            args[last] += (lines[i].strip() + " ")
        i += 1
    args[last] = args[last].strip()
    data = {}
    for key, val in args.items():
        data[key] = {"doc": val}
        default_ = re.search(r"\(default: .*\)",val)
        if default_ is not None:
            default_ = default_.group(0)[1:-1].split(": ")
            assert len(default_) == 2
            default_ = eval(default_[1])
            data[key]["default"] = default_
            doc = re.split(r"\(default: .*\)",val)
            assert len(doc) == 2
            assert doc[1].strip() == ""
            data[key]["doc"] = doc[0].strip()

    return data

class _simple_stdout_filter(object):
    def filter(self, record):
        # only show WARNING or below
        return record.levelno <= logging.WARNING

class _simple_stderr_filter(object):
    def filter(self, record):
        # only show ERROR or above
        return record.levelno >= logging.ERROR

def get_simple_logger(filename=None,
                      stream=None,
                      console=True,
                      level=logging.INFO,
                      formatter=None):
    """Creates a logging object configured to write to any
    combination of a file, a stream, and the console, or
    hide all output.

    Parameters
    ----------
    filename : string, optional
        The name of a file to write to. (default: None)
    stream : file-like object, optional
        A file-like object to write to. (default: None)
    console : bool, optional
        If True, the logger will be configured to print
        output to the console through stdout and
        stderr. (default: True)
    level : int, optional
        The logging level to use. (default: ``logging.INFO``)
    formatter: ``logging.Formatter``, optional
        The logging formatter to use. (default: None)

    Returns
    -------
    ``logging.Logger``
        A logging object
    """
    log = logging.Logger(None, level=level)
    if filename is not None:
        # create file handler which logs even debug messages
        fh = logging.FileHandler(filename)
        fh.setLevel(level)
        log.addHandler(fh)
    if stream is not None:
        ch = logging.StreamHandler(stream)
        ch.setLevel(level)
        log.addHandler(ch)
    if console:
        import sys
        cout = logging.StreamHandler(sys.stdout)
        cout.setLevel(level)
        cout.addFilter(_simple_stdout_filter())
        log.addHandler(cout)
        cerr = logging.StreamHandler(sys.stderr)
        cerr.setLevel(level)
        cerr.addFilter(_simple_stderr_filter())
        log.addHandler(cerr)
    if formatter is not None:
        for h in log.handlers:
            h.setFormatter(formatter)
    if (filename is None) and \
       (stream is None) and \
       (not console):
        log.disabled = True
    return log

def _run_command_line_solver(problem, args):
    import pybnb
    solve_kwds = dict(vars(args))
    del solve_kwds["disable_mpi"]
    del solve_kwds["profile"]
    if args.disable_mpi:
        results = pybnb.solve(problem, comm=None, **solve_kwds)
    else:
        results = pybnb.solve(problem, **solve_kwds)
    return results

def create_command_line_solver(problem, parser=None):
    """Convert a given problem implementation to a
    command-line example by exposing the
    :func:`pybnb.solver.solve` function arguments using
    argparse."""
    import os
    import tempfile
    # for profiling
    try:
        import cProfile as profile
    except ImportError:                                #pragma:nocover
        import profile
    try:
        import pstats
        pstats_available=True
    except ImportError:                                #pragma:nocover
        pstats_available=False
    import pybnb

    if parser is None:
        import argparse
        parser = argparse.ArgumentParser(
            description="Run parallel branch and bound",
            formatter_class=argparse.\
                ArgumentDefaultsHelpFormatter)

    solve_defaults = get_default_args(
        pybnb.Solver.solve)
    solve_docs = get_keyword_docs(
        pybnb.Solver.solve.__doc__)
    solve_docs.pop("problem")
    assert set(solve_defaults.keys()) == \
        set(solve_docs.keys())
    solve_defaults.pop("best_objective")
    solve_docs.pop("best_objective")
    solve_defaults.pop("initialize_queue")
    solve_docs.pop("initialize_queue")
    solve_defaults.pop("log")
    solve_docs.pop("log")
    assert len(solve_defaults) == len(solve_docs)
    for key in solve_defaults:
        assert solve_defaults[key] == \
            solve_docs[key]["default"]
    parser.add_argument(
        "--node-priority-strategy",
        type=str,
        default=solve_defaults.pop("node_priority_strategy"),
        help=solve_docs["absolute_gap"]["doc"])
    parser.add_argument(
        "--absolute-gap",
        type=float,
        default=solve_defaults.pop("absolute_gap"),
        help=solve_docs["absolute_gap"]["doc"])
    parser.add_argument(
        "--relative-gap",
        type=float,
        default=solve_defaults.pop("relative_gap"),
        help=solve_docs["relative_gap"]["doc"])
    parser.add_argument(
        "--cutoff",
        type=float,
        default=solve_defaults.pop("cutoff"),
        help=solve_docs["cutoff"]["doc"])
    parser.add_argument(
        "--node-limit",
        type=int,
        default=solve_defaults.pop("node_limit"),
        help=solve_docs["node_limit"]["doc"])
    parser.add_argument(
        "--time-limit",
        type=float,
        default=solve_defaults.pop("time_limit"),
        help=solve_docs["time_limit"]["doc"])
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=solve_defaults.pop("absolute_tolerance"),
        help=solve_docs["absolute_tolerance"]["doc"])
    parser.add_argument(
        "--log-interval-seconds",
        type=float,
        default=solve_defaults.pop("log_interval_seconds"),
        help=solve_docs["log_interval_seconds"]["doc"])
    assert len(solve_defaults) == 0, str(solve_defaults)

    parser.add_argument(
        "--log-filename", type=str, default=None,
        help=("A filename to store solver output into."))
    parser.add_argument(
        "--results-filename", type=str, default=None,
        help=("When set, saves the solver results into a "
              "YAML-formatted file with the given name."))
    parser.add_argument(
        "--disable-mpi", default=False,
        action="store_true",
        help=("Do not attempt to import mpi4py.MPI."))
    if pstats_available:
        parser.add_argument(
            "--profile", dest="profile", type=int, default=0,
            help=("Enable profiling by setting this "
                  "option to a positive integer (the "
                  "maximum number of functions to "
                  "profile)."))
    args = parser.parse_args()

    if pstats_available and (args.profile):            #pragma:nocover
        #
        # Call the main routine with profiling.
        #
        handle, tfile = tempfile.mkstemp()
        os.close(handle)
        try:
            profile.runctx("_run_command_line_solver(problem, args)",
                           globals(),
                           locals(),
                           tfile)
            p = pstats.Stats(tfile).strip_dirs()
            p.sort_stats("time", "cumulative")
            p = p.print_stats(args.profile)
            p.print_callers(args.profile)
            p.print_callees(args.profile)
            p = p.sort_stats("cumulative","calls")
            p.print_stats(args.profile)
            p.print_callers(args.profile)
            p.print_callees(args.profile)
            p = p.sort_stats("calls")
            p.print_stats(args.profile)
            p.print_callers(args.profile)
            p.print_callees(args.profile)
        finally:
            os.remove(tfile)
    else:
        _run_command_line_solver(problem, args)
