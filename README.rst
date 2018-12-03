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
  Starting branch & bound solve:
   - worker processes: 3
   - node priority strategy: bound
  -----------------------------------------------------------------------------------------------------------------
           Nodes        |                     Objective Bounds                      |              Work
        Expl    Unexpl  |      Incumbent           Bound   Rel. Gap        Abs. Gap |  Time (s)  Nodes/Sec  Starved
           0         1  |            inf            -inf        inf%            inf |      0.00       0.00        0
  *        1         2  |              1              -1    200.000%              2 |      0.00    1239.09        1
  *        2         3  |            0.5              -1    150.000%            1.5 |      0.00    2152.47        0
  *        4         5  |           0.25           -0.25     50.000%            0.5 |      0.00    4736.22        0
  *        8         9  |          0.125         -0.0625     18.750%         0.1875 |      0.00    8724.52        0
        Expl    Unexpl  |      Incumbent           Bound   Rel. Gap        Abs. Gap |  Time (s)  Nodes/Sec  Starved
  *       16        17  |         0.0625       -0.015625      7.812%       0.078125 |      0.00   12643.45        0
  *       32        33  |        0.03125     -0.00390625      3.516%     0.03515625 |      0.00   15273.75        0
  *       64        65  |       0.015625   -0.0009765625      1.660%   0.0166015625 |      0.00   19025.45        0
  *      128       129  |      0.0078125   -0.0002441406      0.806%  0.00805664062 |      0.01   22225.74        0
  *      256       257  |     0.00390625   -6.103516e-05      0.397%  0.00396728516 |      0.01   21489.27        0
        Expl    Unexpl  |      Incumbent           Bound   Rel. Gap        Abs. Gap |  Time (s)  Nodes/Sec  Starved
  *      512       513  |    0.001953125   -1.525879e-05      0.197%  0.00196838379 |      0.02   22939.41        0
  *     1024      1025  |   0.0009765625   -3.814697e-06      0.098% 0.000980377197 |      0.05   20801.99        0
  *     2048      2049  |   0.0004882812   -9.536743e-07      0.049% 0.000489234924 |      0.10   20178.22        0
  *     4096      4097  |   0.0002441406   -2.384186e-07      0.024% 0.000244379044 |      0.20   20557.40        0
  *     8192      8193  |   0.0001220703   -5.960464e-08      0.012% 0.000122129917 |      0.39   20993.97        0
        Expl    Unexpl  |      Incumbent           Bound   Rel. Gap        Abs. Gap |  Time (s)  Nodes/Sec  Starved
  *    16384     16385  |   6.103516e-05   -1.490116e-08      0.006% 6.10500574e-05 |      0.78   20988.63        1
       16386     16387  |   6.103516e-05   -1.490116e-08      0.006% 6.10500574e-05 |      0.78    5476.30        0
  -----------------------------------------------------------------------------------------------------------------
  
  Relative optimality tolerance met
  Optimal solution found
   - absolute gap: 6.10501e-05
   - relative gap: 6.10501e-05
  
  solver results:
   - solution_status: optimal
   - termination_condition: optimality
   - objective: 6.103516e-05
   - bound: -1.490116e-08
   - absolute_gap: 6.105006e-05
   - relative_gap: 6.105006e-05
   - nodes: 16386
   - wall_time: 784.6 ms


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
