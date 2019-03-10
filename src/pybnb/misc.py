"""
Miscellaneous utilities used for development.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""
import logging
import signal
import numbers

def _cast_to_float_or_int(x):
    """Casts a number to a float or int built-in type. Makes
    a reasonable attempt to preserve integrality."""
    if type(x) in (float, int):
        return x
    elif isinstance(x, numbers.Integral):
        x_ = int(x)
        assert x_ == x
        return x_
    else:
        x_ = float(x)
        assert x_ == x
        return x_

class MPI_InterruptHandler(object):
    """A context manager for temporarily assigning a handler
    to SIGINT and SIGUSR1, depending on the availability of
    these signals in the current OS."""
    _sigs = [signal.SIGINT]
    if hasattr(signal, 'SIGUSR1'):
        # not available on windows
        _sigs.append(signal.SIGUSR1)
    __slots__ = ("_released",
                 "_original_handlers",
                 "_handler",
                 "_disable")
    def __init__(self, handler, disable=False):
        self._released = True
        self._original_handlers = None
        self._handler = handler
        self._disable = disable

    def __enter__(self):
        if self._disable:
            return self
        self._released = False
        self._original_handlers = \
            [(signum, signal.getsignal(signum))
             for signum in self._sigs]
        def handler(signum, frame):
            self._handler(signum, frame)
            self.release()
        for signum in self._sigs:
            signal.signal(signum, handler)
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if not self._released:
            for signum, handler in self._original_handlers:
                signal.signal(signum, handler)
            self._released = True

def metric_format(num, unit="s", digits=1, align_unit=False):
    """Format and scale output with metric prefixes.

    Example
    -------

    >>> metric_format(0)
    '0.0 s'
    >>> metric_format(0, align_unit=True)
    '0.0 s '
    >>> metric_format(0.002, unit='B')
    '2.0 mB'
    >>> metric_format(2001, unit='B')
    '2.0 KB'
    >>> metric_format(2001, unit='B', digits=3)
    '2.001 KB'

    """
    if num is None:
        return "<unknown>"
    prefix = ""
    if (num >= 1.0) or (num == 0.0):
        if num >= 1000.0:
            num /= 1000.0
            for p in ['K','M','G','T','P','E','Z','Yi']:
                prefix = p
                if abs(num) < 1000.0:
                    break
                num /= 1000.0
    else:
        num *= 1000.0
        for p in ['m','u','n','p','f']:
            prefix = p
            if abs(num) > 1:
                break
            num *= 1000.0
    if (prefix == "") and align_unit:
        return ("%."+str(digits)+"f %s ") % (num, unit)
    else:
        return ("%."+str(digits)+"f %s%s") % (num, prefix, unit)

def time_format(num, digits=1, align_unit=False):
    """Format and scale output according to standard time
    units.

    Example
    -------

    >>> time_format(0)
    '0.0 s'
    >>> time_format(0, align_unit=True)
    '0.0 s '
    >>> time_format(0.002)
    '2.0 ms'
    >>> time_format(2001)
    '33.4 m'
    >>> time_format(2001, digits=3)
    '33.350 m'

    """
    if num is None:
        return "<unknown>"
    unit = "s"
    if (num >= 1.0) or (num == 0.0):
        if num >= 60.0:
            num /= 60.0
            unit = "m"
            if num >= 60.0:
                num /= 60.0
                unit = "h"
                if num >= 24.0:
                    num /= 24.0
                    unit = "d"
    else:
        num *= 1000.0
        for p in ['ms','us','ns','ps','fs']:
            unit = p
            if abs(num) > 1:
                break
            num *= 1000.0
    if (len(unit) == 1) and align_unit:
        return ("%."+str(digits)+"f %s ") % (num, unit)
    else:
        return ("%."+str(digits)+"f %s") % (num, unit)

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
    return gap_length, gap_label_str, gap_number_str

class _NullCM(object):
    """A context manager that does nothing"""
    def __init__(self, obj):
        self.obj = obj
    def __enter__(self):
        return self.obj
    def __exit__(self, *args):
        pass

def as_stream(stream,
              mode="w",
              **kwds):
    """A utility for handling function arguments that can be
    a filename or a file object. This function is meant to be
    used in the context of a with statement.

    Parameters
    ----------
    stream : file-like object or string
        An existing file-like object or the name of a file
        to open.
    mode : string
        Assigned to the mode keyword of the built-in
        function ``open`` when the `stream` argument is a
        filename. (default: "w")
    **kwds
        Additional keywords passed to the built-in function
        ``open`` when the `stream` argument is a filename.

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
        return open(stream, mode=mode, **kwds)
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
    choices = {}
    i = i_start
    while i != i_stop:
        if i == i_start:
            assert re.match(r".+ : .+(, optional)?", lines[i])
        if re.match(r".+ : .+(, optional)?", lines[i]):
            if i != i_start:
                args[last] = args[last].strip()
            last = lines[i].split(' : ')[0].strip()
            if re.match(r".+ : \{.+\}(, optional)?", lines[i]):
                assert lines[i].count("{") == 1
                assert lines[i].count("}") == 1
                opts = lines[i].split("{")[1].split("}")[0]
                opts = [eval(c.strip()) for c in opts.split(",")]
                choices[last] = opts
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
        if key in choices:
            data[key]["choices"] = choices[key]

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
    if args.nested_solver:
        problem = pybnb.futures.NestedSolver(
            problem,
            node_limit=args.nested_node_limit,
            time_limit=args.nested_time_limit,
            queue_limit=args.nested_queue_limit,
            track_bound=args.nested_track_bound,
            queue_strategy=args.nested_queue_strategy)
    else:
        nested_solver_defaults = get_default_args(
            pybnb.futures.NestedSolver.__init__)
        if args.nested_node_limit != \
           nested_solver_defaults["node_limit"]:     #pragma:nocover
            logging.getLogger("pybnb").warning(
                "The user-specified --nested-node-limit "
                "setting will be ignored. Did you forget the "
                "--nested-solver flag?")
        if args.nested_time_limit != \
           nested_solver_defaults["time_limit"]:     #pragma:nocover
            logging.getLogger("pybnb").warning(
                "The user-specified --nested-time-limit "
                "setting will be ignored. Did you forget the "
                "--nested-solver flag?")
        if args.nested_queue_limit != \
           nested_solver_defaults["queue_limit"]:     #pragma:nocover
            logging.getLogger("pybnb").warning(
                "The user-specified --nested-queue-limit "
                "setting will be ignored. Did you forget the "
                "--nested-solver flag?")
        if args.nested_track_bound != \
           nested_solver_defaults["track_bound"]:     #pragma:nocover
            logging.getLogger("pybnb").warning(
                "The user-specified --nested-disable-track-bound "
                "setting will be ignored. Did you forget the "
                "--nested-solver flag?")
        if args.nested_queue_strategy != \
           nested_solver_defaults["queue_strategy"]: #pragma:nocover
            logging.getLogger("pybnb").warning(
                "The user-specified --nested-queue-strategy "
                "setting will be ignored. Did you forget the "
                "--nested-solver flag?")
    solve_kwds = dict(vars(args))
    del solve_kwds["disable_mpi"]
    del solve_kwds["profile"]
    del solve_kwds["nested_solver"]
    del solve_kwds["nested_node_limit"]
    del solve_kwds["nested_time_limit"]
    del solve_kwds["nested_queue_limit"]
    del solve_kwds["nested_track_bound"]
    del solve_kwds["nested_queue_strategy"]
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
    try:
        import yaml
    except ImportError:                                #pragma:nocover
        raise ImportError("The PyYAML module is required to "
                          "run the command-line solver.")
    from pybnb.convergence_checker import \
        _auto_queue_tolerance
    if parser is None:
        import argparse
        parser = argparse.ArgumentParser(
            description="Run parallel branch and bound",
            formatter_class=argparse.\
                ArgumentDefaultsHelpFormatter)

    solver_init_defaults = get_default_args(
        pybnb.Solver.__init__)
    solver_init_docs = get_keyword_docs(
        pybnb.Solver.__doc__)
    assert set(solver_init_defaults.keys()) == \
        set(solver_init_docs.keys())
    solver_init_defaults.pop("comm")
    solver_init_docs.pop("comm")
    assert len(solver_init_defaults) == len(solver_init_docs)
    for key in solver_init_defaults:
        assert solver_init_defaults[key] == \
            solver_init_docs[key]["default"]
        assert "choices" not in solver_init_docs[key]
    parser.add_argument(
        "--dispatcher-rank",
        type=int,
        default=solver_init_defaults.pop("dispatcher_rank"),
        help=solver_init_docs["dispatcher_rank"]["doc"])
    assert len(solver_init_defaults) == 0, str(solver_init_defaults)

    solve_defaults = get_default_args(
        pybnb.Solver.solve)
    solve_docs = get_keyword_docs(
        pybnb.Solver.solve.__doc__)
    solve_docs.pop("problem")
    assert set(solve_defaults.keys()) == \
        set(solve_docs.keys())
    solve_defaults.pop("best_node")
    solve_docs.pop("best_node")
    solve_defaults.pop("initialize_queue")
    solve_docs.pop("initialize_queue")
    solve_defaults.pop("scale_function")
    solve_docs.pop("scale_function")
    solve_defaults.pop("log")
    solve_docs.pop("log")
    assert len(solve_defaults) == len(solve_docs)
    for key in solve_defaults:
        if key == "queue_tolerance":
            assert "default" not in solve_docs[key]
            assert solve_defaults[key] is \
                _auto_queue_tolerance
        else:
            assert solve_defaults[key] == \
                solve_docs[key]["default"]
        assert "choices" not in solve_docs[key]
        if key == "queue_strategy":
            solve_docs[key]["choices"] = \
                [v_.value for v_ in pybnb.QueueStrategy]
            assert "**(D)**" in solve_docs[key]["doc"]
            solve_docs[key]["doc"] = \
                ("**(D)** Sets the strategy for prioritizing "
                 "nodes in the central dispatcher queue. Can "
                 "also be set to a comma-separated list of "
                 "choices to define a lexicographic sorting "
                 "strategy.")
    class _QueueStrategyJoin(argparse.Action):    #pragma:nocover
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            if nargs is not None:
                raise ValueError("nargs not allowed")
            super(_QueueStrategyJoin, self).__init__(option_strings,
                                                     dest,
                                                     **kwargs)
        def __call__(self, parser, namespace, values, option_string=None):
            if values in solve_docs["queue_strategy"]["choices"]:
                namespace.queue_strategy = values
            else:
                assert "," in values
                vals = tuple(v.strip() for v in values.split(',') if v.strip())
                assert len(vals) > 0
                assert all(v in solve_docs["queue_strategy"]["choices"]
                           for v in vals)
                namespace.queue_strategy = vals
    class _QueueStrategyChoices(object):          #pragma:nocover
        def __contains__(self, val):
            if val in solve_docs["queue_strategy"]["choices"]:
                return True
            if "," not in val:
                return False
            vals = [v.strip() for v in val.split(',') if v.strip()]
            if len(vals) == 0:
                return False
            return all(v in solve_docs["queue_strategy"]["choices"]
                       for v in vals)
        def __iter__(self):
            return solve_docs["queue_strategy"]["choices"].__iter__()
    parser.add_argument(
        "--best-objective",
        type=float,
        default=solve_defaults.pop("best_objective"),
        help=solve_docs["best_objective"]["doc"])
    tmp_ = solve_defaults.pop("disable_objective_call")
    assert not tmp_
    del tmp_
    parser.add_argument(
        "--disable-objective-call",
        default=False,
        action="store_true",
        help=solve_docs["disable_objective_call"]["doc"])
    parser.add_argument(
        "--queue-strategy",
        type=str,
        choices=_QueueStrategyChoices(),
        action=_QueueStrategyJoin,
        default=solve_defaults.pop("queue_strategy"),
        help=solve_docs["queue_strategy"]["doc"])
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
        "--objective-stop",
        type=float,
        default=solve_defaults.pop("objective_stop"),
        help=solve_docs["objective_stop"]["doc"])
    parser.add_argument(
        "--bound-stop",
        type=float,
        default=solve_defaults.pop("bound_stop"),
        help=solve_docs["bound_stop"]["doc"])
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
        "--queue-limit",
        type=int,
        default=solve_defaults.pop("queue_limit"),
        help=solve_docs["queue_limit"]["doc"])
    val = solve_defaults.pop("track_bound")
    assert val
    parser.add_argument(
        "--disable-track-bound",
        action="store_false",
        dest="track_bound",
        default=True,
        help=solve_docs["track_bound"]["doc"])
    def _float_or_None(val):                      #pragma:nocover
        if val == "None":
            return None
        return float(val)
    parser.add_argument(
        "--queue-tolerance",
        type=_float_or_None,
        default=solve_defaults.pop("queue_tolerance"),
        help=solve_docs["queue_tolerance"]["doc"])
    parser.add_argument(
        "--branch-tolerance",
        type=_float_or_None,
        default=solve_defaults.pop("branch_tolerance"),
        help=solve_docs["branch_tolerance"]["doc"])
    parser.add_argument(
        "--comparison-tolerance",
        type=float,
        default=solve_defaults.pop("comparison_tolerance"),
        help=solve_docs["comparison_tolerance"]["doc"])
    parser.add_argument(
        "--log-interval-seconds",
        type=float,
        default=solve_defaults.pop("log_interval_seconds"),
        help=solve_docs["log_interval_seconds"]["doc"])
    val = solve_defaults.pop("log_new_incumbent")
    assert val
    parser.add_argument(
        "--disable-log-new-incumbent",
        action="store_false",
        dest="log_new_incumbent",
        default=True,
        help=solve_docs["log_new_incumbent"]["doc"])
    val = solve_defaults.pop("disable_signal_handlers")
    assert not val
    parser.add_argument(
        "--disable-signal-handlers",
        action="store_true",
        default=False,
        help=solve_docs["disable_signal_handlers"]["doc"])
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
        help=("Do not attempt to import mpi4py.MPI. "
              "Enabling this option is equivalent to "
              "creating a Solver with `comm=None`."))
    if pstats_available:
        parser.add_argument(
            "--profile", dest="profile", type=int, default=0,
            help=("Enable profiling by setting this "
                  "option to a positive integer (the "
                  "maximum number of functions to "
                  "profile)."))
    parser.add_argument('--version',
                        action='version',
                        version='pybnb '+str(pybnb.__version__))
    nested_solver_defaults = get_default_args(
        pybnb.futures.NestedSolver.__init__)
    nested_solver_docs = get_keyword_docs(
        pybnb.futures.NestedSolver.__doc__)
    nested_solver_docs.pop("problem")
    assert len(nested_solver_defaults) == len(nested_solver_docs)
    parser.add_argument(
        "--nested-solver",
        action="store_true",
        default=False,
        help=("**(W)** Wraps the problem in a "
              ":class:`pybnb.futures.NestedSolver` object. "
              "See additional --nested-solver-* options."))
    parser.add_argument(
        "--nested-node-limit",
        type=int,
        default=nested_solver_defaults.pop("node_limit"),
        help=nested_solver_docs["node_limit"]["doc"])
    parser.add_argument(
        "--nested-time-limit",
        type=float,
        default=nested_solver_defaults.pop("time_limit"),
        help=nested_solver_docs["time_limit"]["doc"])
    parser.add_argument(
        "--nested-queue-limit",
        type=int,
        default=nested_solver_defaults.pop("queue_limit"),
        help=nested_solver_docs["queue_limit"]["doc"])
    val = nested_solver_defaults.pop("track_bound")
    assert val
    parser.add_argument(
        "--nested-disable-track-bound",
        action="store_false",
        dest="nested_track_bound",
        default=True,
        help=nested_solver_docs["track_bound"]["doc"])
    parser.add_argument(
        "--nested-queue-strategy",
        type=str,
        choices=_QueueStrategyChoices(),
        action=_QueueStrategyJoin,
        default=nested_solver_defaults.pop("queue_strategy"),
        help=nested_solver_docs["queue_strategy"]["doc"])
    assert len(nested_solver_defaults) == 0,\
        str(nested_solver_defaults)

    args = parser.parse_args()

    try:
        import mpi4py
    except ImportError:                                #pragma:nocover
        if not args.disable_mpi:
            raise ImportError("The mpi4py module is not "
                              "available. To run this script "
                              "without it, use the '--disable-mpi' "
                              "option")

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
