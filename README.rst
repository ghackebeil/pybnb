pybnb
=====

|PyPI-Status| |PyPI-Versions| |Codacy-Grade|

|Travis-Status| |Appveyor-Status| |Coverage-Status| |Documentation-Status|

A parallel branch-and-bound engine for Python. (https://pybnb.readthedocs.io)

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
          return round(self.bounds[1] - self.bounds[0], 3)
      def bound(self):
          return -(self.bounds[1] - self.bounds[0])**2
      def save_state(self, node):
          node.resize(2)
          node.state[:] = self.bounds
      def load_state(self, node):
          self.bounds = node.state.tolist()
      def branch(self, parent):
          L, U = self.bounds
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
  results = pybnb.solve(problem,
                        relative_gap=1e-9,
                        absolute_gap=1e-9)

**Run the script:**

.. code:: bash

  $ mpirun -np 4 python solve_simple.py
  Starting branch & bound solve:
   - worker processes: 3
   - node priority strategy: bound
  ---------------------------------------------------------------------------------------------------------------------
           Nodes        |                       Objective Bounds                        |              Work
        Expl    Unexpl  |      Incumbent           Bound      Rel. Gap         Abs. Gap | Time (s)  Nodes/Sec Imbalance
           0         1  |            inf            -inf           inf%             inf |      0.0       0.00     0.00%
  *        1         2  |              1              -1  200.00000000%               2 |      0.0     964.32   300.00%
  *        2         3  |            0.5              -1  150.00000000%             1.5 |      0.0     963.67   150.00%
  *        4         5  |           0.25           -0.25   50.00000000%             0.5 |      0.0    2136.25    75.00%
  *        8         9  |          0.125         -0.0625   18.75000000%          0.1875 |      0.0    4693.56    37.50%
        Expl    Unexpl  |      Incumbent           Bound      Rel. Gap         Abs. Gap | Time (s)  Nodes/Sec Imbalance
  *       16        17  |          0.062       -0.015625    7.76250000%        0.077625 |      0.0    8308.76    18.75%
  *       32        33  |          0.031     -0.00390625    3.49062500%      0.03490625 |      0.0   11921.58    28.12%
  *       64        65  |          0.016   -0.0009765625    1.69765625%    0.0169765625 |      0.0   15898.55    18.75%
  *      128       129  |          0.008   -0.0002441406    0.82441406%  0.008244140625 |      0.0   18406.37     9.38%
  *      256       257  |          0.004   -6.103516e-05    0.40610352%  0.004061035156 |      0.0   19013.29     9.38%
        Expl    Unexpl  |      Incumbent           Bound      Rel. Gap         Abs. Gap | Time (s)  Nodes/Sec Imbalance
  *      512       513  |          0.002   -1.525879e-05    0.20152588%  0.002015258789 |      0.0   19864.55     7.03%
  *     1024      1025  |          0.001   -3.814697e-06    0.10038147%  0.001003814697 |      0.1   18221.83     4.10%
  *     2048      2049  |              0   -9.536743e-07    0.00009537% 9.536743164e-07 |      0.1   17859.95     2.49%
       21653     21654  |              0   -1.490116e-08    0.00000149% 1.490116119e-08 |      1.1   19046.61     3.82%
       41037     41038  |              0    -3.72529e-09    0.00000037% 3.725290298e-09 |      2.1   19281.25     4.40%
        Expl    Unexpl  |      Incumbent           Bound      Rel. Gap         Abs. Gap | Time (s)  Nodes/Sec Imbalance
       60400     60401  |              0    -3.72529e-09    0.00000037% 3.725290298e-09 |      3.1   19338.39     4.14%
       65537     65538  |              0   -9.313226e-10    0.00000009% 9.313225746e-10 |      3.4   19149.73     4.39%
  ---------------------------------------------------------------------------------------------------------------------
  
  Absolute optimality tolerance met
  Relative optimality tolerance met
  Optimal solution found
   - absolute gap: 9.31323e-10
   - relative gap: 9.31323e-10
  
  solver results:
   - solution_status: optimal
   - termination_condition: optimality
   - objective: 0
   - bound: -9.313226e-10
   - absolute_gap: 9.313226e-10
   - relative_gap: 9.313226e-10
   - nodes: 65537
   - wall_time: 3.39 s


.. |Travis-Status| image:: https://travis-ci.org/ghackebeil/pybnb.svg?branch=master
  :target: https://travis-ci.org/ghackebeil/pybnb
.. |Appveyor-Status| image::  https://ci.appveyor.com/api/projects/status/gfbrxja9v08rm7a2?svg=true
  :target: https://ci.appveyor.com/project/ghackebeil/pybnb
.. |Coverage-Status| image:: https://codecov.io/gh/ghackebeil/pybnb/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/ghackebeil/pybnb
.. |Documentation-Status| image:: https://readthedocs.org/projects/pybnb/badge/?version=latest
  :target: http://pybnb.readthedocs.io/en/latest/?badge=latest
.. |PyPI-Status| image:: https://img.shields.io/pypi/v/pybnb.svg
  :target: https://pypi.python.org/pypi/pybnb/
.. |PyPI-Versions| image:: https://img.shields.io/pypi/pyversions/pybnb.svg
   :target: https://pypi.org/project/pybnb
.. |Codacy-Grade| image:: https://img.shields.io/lgtm/grade/python/g/ghackebeil/pybnb.svg?logo=lgtm&logoWidth=18
   :target: https://lgtm.com/projects/g/ghackebeil/pybnb/context:python
