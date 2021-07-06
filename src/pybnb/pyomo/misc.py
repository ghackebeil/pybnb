"""
Miscellaneous utilities used for development.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""
import math
import array
import collections
import hashlib

import pybnb

import six
from six.moves import xrange as range

import pyomo.kernel as pmo

if getattr(pmo, "version_info", (0,) * 3) < (5, 4, 3):  # pragma:nocover
    raise ImportError("Pyomo 5.4.3 or later is not available")


def hash_joblist(jobs):
    """Create a hash of a Python list by casting each entry
    to a string."""
    x = hashlib.sha1()
    for entry in jobs:
        x.update(str(entry).encode())
    return x.hexdigest()


def add_tmp_component(model, name, obj):
    """Add a temporary component to a model, adjusting the
    name as needed to make sure it is unique."""
    while hasattr(model, name):
        name = "." + name + "."
    setattr(model, name, obj)
    return name


def correct_integer_lb(lb, integer_tolerance):
    """Converts a lower bound for an integer optimization
    variable to an integer equal to `ceil(ub)`, taking care
    not to move a non-integer bound away from an integer
    point already within a given tolerance."""
    assert 0 <= integer_tolerance < 0.5
    if lb - math.floor(lb) > integer_tolerance:
        return int(math.ceil(lb))
    else:
        return int(math.floor(lb))


def correct_integer_ub(ub, integer_tolerance):
    """Converts an upper bound for an integer optimization
    variable to an integer equal to `floor(ub)`, taking care
    not to move a non-integer bound away from an integer
    point already within a given tolerance."""
    assert 0 <= integer_tolerance < 0.5
    if math.ceil(ub) - ub > integer_tolerance:
        return int(math.floor(ub))
    else:
        return int(math.ceil(ub))


def create_optimality_bound(problem, pyomo_objective, best_objective_value):
    """Returns a constraint that bounds an objective
    function with a known best value. That is, the
    constraint will require the objective function to be
    better than the given value."""
    optbound = pmo.constraint(body=pyomo_objective)
    if problem.sense() == pybnb.minimize:
        assert pyomo_objective.sense == pmo.minimize
        optbound.ub = best_objective_value
    else:
        assert problem.sense() == pybnb.maximize
        assert pyomo_objective.sense == pmo.maximize
        optbound.lb = best_objective_value
    return optbound


def generate_cids(model, prefix=(), **kwds):
    """Generate forward and reverse mappings between model
    components and deterministic, unique identifiers that
    are safe to serialize or use as dictionary keys."""
    object_to_cid = pmo.ComponentMap()
    cid_to_object = collections.OrderedDict()
    if hasattr(pmo, "preorder_traversal"):  # pragma:nocover
        fn = lambda *args, **kwds: pmo.preorder_traversal(model, *args, **kwds)
    else:  # pragma:nocover
        fn = model.preorder_traversal
    try:
        fn(return_key=True)
    except TypeError:
        traversal = fn(**kwds)
        obj_ = six.next(traversal)
        assert obj_ is model
        object_to_cid[model] = prefix
        cid_to_object[prefix] = model
        for obj in traversal:
            parent = obj.parent
            key = obj.storage_key
            cid_ = object_to_cid[obj] = object_to_cid[parent] + (key,)
            cid_to_object[cid_] = obj
    else:  # pragma:nocover
        traversal = fn(return_key=True, **kwds)
        obj_ = six.next(traversal)[1]
        assert obj_ is model
        object_to_cid[model] = prefix
        cid_to_object[prefix] = model
        for key, obj in traversal:
            parent = obj.parent
            cid_ = object_to_cid[obj] = object_to_cid[parent] + (key,)
            cid_to_object[cid_] = obj
    return object_to_cid, cid_to_object
