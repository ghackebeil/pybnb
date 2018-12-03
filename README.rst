pybnb
=====

.. image:: https://travis-ci.org/ghackebeil/pybnb.svg?branch=master
  :target: https://travis-ci.org/ghackebeil/pybnb
.. image::  https://ci.appveyor.com/api/projects/status/gfbrxja9v08rm7a2?svg=true
  :target: https://ci.appveyor.com/project/ghackebeil/pybnb
.. image:: https://codecov.io/gh/ghackebeil/pybnb/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/ghackebeil/pybnb
.. image:: https://img.shields.io/pypi/v/pybnb.svg
  :target: https://pypi.python.org/pypi/pybnb/
.. image:: https://readthedocs.org/projects/pybnb/badge/?version=latest
  :target: http://pybnb.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

A parallel branch-and-bound engine for Python.

This software is copyright (c) by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).

This software is released under the MIT software license.
This license, including disclaimer, is available in the 'LICENSE' file.

Quick Start
-----------

**Define a problem:**

.. code:: python

  # simple.py

  import pybnb

  class Simple(pybnb.Problem):

      def __init__(self):
          self.bounds = [0.0,1.0]
      def sense(self):
          return pybnb.minimize
      def objective(self):
          return self.bounds[1] - self.bounds[0]
      def bound(self):
          return -(self.bounds[0] - self.bounds[1])**2
      def save_state(self, node):
          node.resize(2)
          node.state[:] = self.bounds
      def load_state(self, node):
          self.bounds = node.state.tolist()
      def branch(self, parent):
          L, U = self.bounds
          if U-L <= 1e-8:
              return ()
          mid = 0.5 * (L + U)
          left = parent.new_child()
          left.state[:] = (L, mid)
          right = parent.new_child()
          right.state[:] = (mid, U)
          return left, right

**Write a solve script:**

.. code:: python

  # solve_simple.py

  import simple
  problem = simple.Simple()
  results = pybnb.solve(problem)

**Run the script:**

.. code:: bash

  $ mpirun -np 4 python solve_simple.py
