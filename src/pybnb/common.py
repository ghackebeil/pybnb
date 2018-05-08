"""
Basic definitions and utilities.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""

minimize = 1
"""The objective sense defining a minimization problem."""

maximize = -1
"""The objective sense defining a maximization problem."""

infinity = float("inf")
"""A constant equal to ``float('inf')``."""

def is_infinite(x):
    """Returns True if the given argument is equal to `+inf`
    or `-inf`.

    Example
    -------

    >>> is_infinite(float('inf'))
    True
    >>> is_infinite(float('-inf'))
    True
    >>> is_infinite(0)
    False

    """
    return (x == -infinity) or \
           (x == infinity)
