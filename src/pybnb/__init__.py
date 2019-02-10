
# configure a very basic logger for the module
def _configLogging():
    import logging
    logger = logging.getLogger('pybnb')
    logger.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        '%(levelname)s(%(name)s): %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
_configLogging()
del _configLogging

from pybnb.__about__ import __version__
from pybnb.configuration import config
from pybnb.common import (minimize,
                          maximize,
                          inf,
                          nan,
                          QueueStrategy,
                          SolutionStatus,
                          TerminationCondition)
from pybnb.priority_queue import (CustomPriorityQueue,
                                  register_queue_type)
from pybnb.node import Node
from pybnb.problem import Problem
from pybnb.solver_results import SolverResults
from pybnb.solver import (Solver,
                          solve)
