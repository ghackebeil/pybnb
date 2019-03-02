"""
Branch-and-bound solver results object.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""

# recognized pytest-doctestplus plugin,
# not the standard doctest
__doctest_requires__ = {'SolverResults.write': ['yaml']}

import sys
import base64

from pybnb.common import (SolutionStatus,
                          TerminationCondition)
from pybnb.misc import (time_format,
                        as_stream)
from pybnb.node import dumps

import six

class SolverResults(object):
    """Stores the results of a branch-and-bound solve.

    Attributes
    ----------
    solution_status : string
        The solution status will be set to one of the strings
        documented by the :class:`SolutionStatus
        <pybnb.common.SolutionStatus>` enum.
    termination_condition : string
        The solve termination condition, as determined by
        the dispatcher, will be set to one of the strings
        documented by the :class:`TerminationCondition
        <pybnb.common.TerminationCondition>` enum.
    objective : float
        The best objective found.
    bound : float
        The global optimality bound.
    absolute_gap : float or None
        The absolute gap between the objective and
        bound. This will only be set when the solution
        status sf "optimal" or "feasible"; otherwise, it
        will be None.
    relative_gap : float or None
        The relative gap between the objective and
        bound. This will only be set when the solution
        status sf "optimal" or "feasible"; otherwise, it
        will be None.
    nodes : int
        The total number of nodes processes by all workers.
    wall_time : float
        The process-local wall time (seconds). This is the
        only value on the results object that varies between
        processes.
    best_node : :class:`Node <pybnb.node.Node>`
        The node with the best objective obtained during the
        solve. Note that if the best_objective solver option
        was used, the best_node on the results object may
        have an objective that is worse than the objective
        stored on the results (or may be None).
    """

    def __init__(self):
        self.solution_status = None
        self.termination_condition = None
        self.objective = None
        self.bound = None
        self.absolute_gap = None
        self.relative_gap = None
        self.nodes = None
        self.wall_time = None
        self.best_node = None

    def pprint(self, stream=sys.stdout):
        """Prints a nicely formatted representation of the
        results.

        Parameters
        ----------
        stream : file-like object or string, optional
            A file-like object or a filename where results
            should be written to. (default: ``sys.stdout``)
        """
        with as_stream(stream) as stream:
            stream.write("solver results:\n")
            self.write(stream, prefix=" - ", pretty=True)

    def write(self, stream, prefix="", pretty=False):
        """Writes results in YAML format to a stream or
        file. Changing the parameter values from their
        defaults may result in the output becoming
        non-compatible with the YAML format.

        Parameters
        ----------
        stream : file-like object or string
            A file-like object or a filename where results
            should be written to.
        prefix : string, optional
            A string to use as a prefix for each line that
            is written. (default: '')
        pretty : bool, optional
            Indicates whether or not certain recognized
            attributes should be formatted for more
            human-readable output. (default: False)

        Example
        -------

        >>> import six
        >>> import yaml
        >>> import pybnb
        >>> results = pybnb.SolverResults()
        >>> results.best_node = pybnb.Node()
        >>> results.best_node.objective = 123
        >>> out = six.StringIO()
        >>> # the best_node is serialized
        >>> results.write(out)
        >>> del results
        >>> results_dict = yaml.load(out.getvalue())
        >>> # de-serialize the best_node
        >>> best_node = pybnb.node.loads(results_dict['best_node'])
        >>> assert best_node.objective == 123

        """
        with as_stream(stream) as stream:
            attrs = vars(self)
            names = sorted(list(attrs.keys()))
            first = ('solution_status', 'termination_condition',
                     'objective', 'bound',
                     'absolute_gap', 'relative_gap',
                     'nodes', 'wall_time', 'best_node')
            for cnt, name in enumerate(first):
                if not hasattr(self, name):
                    continue
                names.remove(name)
                val = getattr(self, name)
                if val is not None:
                    if name in ("solution_status",
                                "termination_condition"):
                        if type(val) in (SolutionStatus,
                                         TerminationCondition):
                            val = val.value
                    elif pretty:
                        if name == 'wall_time':
                            val = time_format(val,
                                              digits=2)
                        elif name in ('objective',
                                      'bound',
                                      'absolute_gap',
                                      'relative_gap'):
                            val = "%.7g" % (val)
                        elif name == "best_node":
                            if val.objective is not None:
                                val = ("Node(objective=%.7g)"
                                       % (val.objective))
                            else:
                                val = "Node(objective=None)"
                    else:
                        if name == "best_node":
                            val = dumps(val)
                            if hasattr(base64, 'encodebytes'):
                                val = base64.encodebytes(val).\
                                    decode("ascii")
                            else:
                                val = base64.encodestring(val).\
                                    decode("ascii")
                            val = ('\n  '.join(
                                val.splitlines()))
                            val = ("!!binary |\n  %s"
                                   % (val))
                        else:
                            val_ = "%r" % (val)
                            if type(val) is float:
                                if val_ == 'inf':
                                    val_ = '.inf'
                                elif val_ == '-inf':
                                    val_ = "-.inf"
                                elif val_ == 'nan':
                                    val_ = ".nan"
                            val = val_
                            del val_
                if pretty or (val is not None):
                    stream.write(prefix+'%s: %s\n'
                                 % (name, val))
                else:
                    assert val is None
                    stream.write(prefix+'%s: null\n'
                                 % (name))
            for name in names:
                val = getattr(self, name)
                if pretty:
                    stream.write(prefix+'%s: %r\n'
                                 % (name, val))
                else:
                    if val is None:
                        stream.write(prefix+'%s: null\n'
                                     % (name))
                    else:
                        val_ = "%r" % (val)
                        if type(val) is float:
                            if val_ == 'inf':
                                val_ = '.inf'
                            elif val_ == '-inf':
                                val_ = "-.inf"
                            elif val_ == 'nan':
                                val_ = ".nan"
                        val = val_
                        del val_
                        stream.write(prefix+'%s: %s\n'
                                     % (name, val))

    def __str__(self):
        """Represents the results as a string."""
        tmp = six.StringIO()
        self.pprint(stream=tmp)
        return tmp.getvalue()
