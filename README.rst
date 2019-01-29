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
  
  Using non-default solver options:
   - absolute_gap: 1e-09 (default: 1e-08)
   - relative_gap: 1e-09 (default: 0.0001)
  
  Starting branch & bound solve:
   - dispatcher pid: 45083 (Ozymandias.local)
   - worker processes: 3
  ----------------------------------------------------------------------------------------------------------------------------
           Nodes        |                       Objective Bounds                        |              Work                   
        Expl    Unexpl  |      Incumbent           Bound      Rel. Gap         Abs. Gap | Time (s)  Nodes/Sec Imbalance   Idle
           0         1  |            inf            -inf           inf%             inf |      0.0       0.00     0.00%      0
  *        1         2  |              1              -1  200.00000000%               2 |      0.0    1141.55   300.00%      1
  *        2         3  |            0.5              -1  150.00000000%             1.5 |      0.0    2302.82   150.00%      0
  *        4         5  |           0.25           -0.25   50.00000000%             0.5 |      0.0    8363.83    75.00%      0
  *        8         9  |          0.125         -0.0625   18.75000000%          0.1875 |      0.0   13275.25    37.50%      0
        Expl    Unexpl  |      Incumbent           Bound      Rel. Gap         Abs. Gap | Time (s)  Nodes/Sec Imbalance   Idle
  *       16        17  |          0.062       -0.015625    7.76250000%        0.077625 |      0.0   16424.36    18.75%      0
  *       32        33  |          0.031     -0.00390625    3.49062500%      0.03490625 |      0.0   18441.51    18.75%      0
  *       64        65  |          0.016   -0.0009765625    1.69765625%    0.0169765625 |      0.0   20585.25    28.12%      0
  *      128       129  |          0.008   -0.0002441406    0.82441406%  0.008244140625 |      0.0   21657.47    16.41%      0
  *      256       257  |          0.004   -6.103516e-05    0.40610352%  0.004061035156 |      0.0   21710.52    15.23%      0
        Expl    Unexpl  |      Incumbent           Bound      Rel. Gap         Abs. Gap | Time (s)  Nodes/Sec Imbalance   Idle
  *      511       512  |          0.002   -6.103516e-05    0.20610352%  0.002061035156 |      0.0   21037.48    12.92%      0
  *     1024      1025  |          0.001   -3.814697e-06    0.10038147%  0.001003814697 |      0.1   18367.08    14.65%      0
  *     2048      2049  |              0   -9.536743e-07    0.00009537% 9.536743164e-07 |      0.1   16960.48    17.14%      0
       22335     22336  |              0   -1.490116e-08    0.00000149% 1.490116119e-08 |      1.1   20089.97    14.80%      0
       42650     42651  |              0    -3.72529e-09    0.00000037% 3.725290298e-09 |      2.1   20303.24    13.70%      0
        Expl    Unexpl  |      Incumbent           Bound      Rel. Gap         Abs. Gap | Time (s)  Nodes/Sec Imbalance   Idle
       63908     63909  |              0    -3.72529e-09    0.00000037% 3.725290298e-09 |      3.1   21207.89    13.21%      0
       65537     65538  |              0   -9.313226e-10    0.00000009% 9.313225746e-10 |      3.2   19857.83    13.24%      0
  ----------------------------------------------------------------------------------------------------------------------------
  
  Absolute optimality tolerance met
  Relative optimality tolerance met
  Optimal solution found!
  
  solver results:
   - solution_status: optimal
   - termination_condition: optimality
   - objective: 0
   - bound: -9.313226e-10
   - absolute_gap: 9.313226e-10
   - relative_gap: 9.313226e-10
   - nodes: 65537
   - wall_time: 3.20 s


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
