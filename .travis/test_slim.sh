#!/bin/bash

set -e

${DOC} pip uninstall -y mpi4py
${DOC} pip uninstall -y Pyomo
${DOC} pip uninstall -y Pyutilib
${DOC} pip uninstall -y numba
${DOC} pytest -v --cov=pybnb --cov=examples --cov=src/tests --cov-report="" --run-examples
${DOC} mv .coverage coverage.slim.1
${DOC} pip install Pyomo
${DOC} pytest -v --cov=pybnb --cov=examples --cov=src/tests --cov-report="" --run-examples
${DOC} mv .coverage coverage.slim.2
# prepare the coverage files for the "coverage combine"
# call that comes next
${DOC} mv coverage.slim.1 .coverage.slim.1
${DOC} mv coverage.slim.2 .coverage.slim.2
