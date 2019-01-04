"""
A Base class for defining a branch-and-bound problem based
on a pyomo.kernel model.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""
from pybnb import Problem
from pybnb.pyomo.misc import generate_cids

class PyomoProblem(Problem):
    """An extension of the :class:`pybnb.Problem
    <pybnb.problem.Problem>` base class for defining
    problems with a core Pyomo model."""

    def __init__(self, *args, **kwds):
        super(PyomoProblem, self).__init__(*args, **kwds)
        self.__pyomo_object_to_cid = None
        self.__cid_to_pyomo_object = None
        self.update_pyomo_object_cids()

    def update_pyomo_object_cids(self):
        (self.__pyomo_object_to_cid,
         self.__cid_to_pyomo_object) = \
            generate_cids(self.pyomo_model,
                          active=None)
        assert len(set(self.pyomo_object_to_cid.values())) == \
            len(self.__pyomo_object_to_cid)
        assert len(self.cid_to_pyomo_object) == \
            len(self.__pyomo_object_to_cid)

    @property
    def pyomo_object_to_cid(self):
        """The map from pyomo model object to component id."""
        return self.__pyomo_object_to_cid

    @property
    def cid_to_pyomo_object(self):
        """The map from component id to pyomo model object."""
        return self.__cid_to_pyomo_object

    #
    # Abstract Methods
    #

    @property
    def pyomo_model(self):
        """Returns the pyomo model for this problem."""
        raise NotImplementedError()               #pragma:nocover

    @property
    def pyomo_model_objective(self):
        """Returns the pyomo model objective for this
        problem."""
        raise NotImplementedError()               #pragma:nocover
