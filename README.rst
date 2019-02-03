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
          node.state = self.bounds
      def load_state(self, node):
          self.bounds = node.state
      def branch(self):
          L, U = self.bounds
          mid = 0.5 * (L + U)
          for l,u in [(L,mid), (mid,U)]:
              child = pybnb.Node()
              child.state = (l,u)
              yield child

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
   - dispatcher pid: 48432 (Ozymandias.local)
   - worker processes: 3
  ----------------------------------------------------------------------------------------------------------------------------
           Nodes        |                       Objective Bounds                        |              Work              
        Expl    Unexpl  |      Incumbent           Bound      Rel. Gap         Abs. Gap | Time (s)  Nodes/Sec Imbalance   Idle
           0         1  |            inf            -inf           inf%             inf |      0.0       0.00     0.00%      0
  *        1         2  |              1              -1  200.00000000%               2 |      0.0     905.80   300.00%      1
  *        2         3  |            0.5              -1  150.00000000%             1.5 |      0.0    1547.75   150.00%      0
  *        4         5  |           0.25           -0.25   50.00000000%             0.5 |      0.0    5751.92    75.00%      0
  *        8         9  |          0.125         -0.0625   18.75000000%          0.1875 |      0.0   12931.55    37.50%      0
        Expl    Unexpl  |      Incumbent           Bound      Rel. Gap         Abs. Gap | Time (s)  Nodes/Sec Imbalance   Idle
  *       16        17  |          0.062       -0.015625    7.76250000%        0.077625 |      0.0   16429.40    18.75%      0
  *       32        33  |          0.031     -0.00390625    3.49062500%      0.03490625 |      0.0   13745.78    18.75%      0
  *       64        65  |          0.016   -0.0009765625    1.69765625%    0.0169765625 |      0.0    9441.90    14.06%      0
  *      128       129  |          0.008   -0.0002441406    0.82441406%  0.008244140625 |      0.0   13021.19    11.72%      0
  *      256       257  |          0.004   -6.103516e-05    0.40610352%  0.004061035156 |      0.0   14889.08    15.23%      0
        Expl    Unexpl  |      Incumbent           Bound      Rel. Gap         Abs. Gap | Time (s)  Nodes/Sec Imbalance   Idle
  *      512       513  |          0.002   -1.525879e-05    0.20152588%  0.002015258789 |      0.0   11448.97    16.41%      0
  *     1024      1025  |          0.001   -3.814697e-06    0.10038147%  0.001003814697 |      0.1   12000.69    14.06%      0
  *     2048      2049  |              0   -9.536743e-07    0.00009537% 9.536743164e-07 |      0.2   12120.58    13.33%      0
       17167     17168  |              0   -1.490116e-08    0.00000149% 1.490116119e-08 |      1.2   14933.56    12.49%      0
       31536     31537  |              0   -1.490116e-08    0.00000149% 1.490116119e-08 |      2.2   14396.01    12.60%      0
        Expl    Unexpl  |      Incumbent           Bound      Rel. Gap         Abs. Gap | Time (s)  Nodes/Sec Imbalance   Idle
       47116     47117  |              0    -3.72529e-09    0.00000037% 3.725290298e-09 |      3.2   15515.93    13.47%      0
       63124     63125  |              0    -3.72529e-09    0.00000037% 3.725290298e-09 |      4.2   15982.17    12.50%      0
       65537     65538  |              0   -9.313226e-10    0.00000009% 9.313225746e-10 |      4.3   15622.95    12.47%      0
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
   - wall_time: 4.33 s
  
  Number of Workers:        3
  Load Imbalance:      12.47%
   - min: 20632
   - max: 23357
  Average Worker Timing:
   - queue:      84.77% [avg time: 168.4 us, count: 65537]
   - load_state:  0.50% [avg time: 996.5 ns, count: 65537]
   - bound:       0.61% [avg time:   1.2 us, count: 65537]
   - objective:   1.68% [avg time:   3.3 us, count: 65537]
   - branch:      3.96% [avg time:   7.8 us, count: 65537]
   - other:       8.48% [avg time:  16.8 us, count: 65537]


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
