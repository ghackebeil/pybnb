#!/bin/bash

set -e

${DOC_} pip uninstall -y mpi4py
${DOC_} pip uninstall -y Pyomo
${DOC_} pip uninstall -y Pyutilib
${DOC_} pip uninstall -y numba
${DOC_} pytest -v --cov=pybnb --cov=examples --cov=src/tests --cov-report="" --run-examples
${DOC_} mv .coverage .coverage.slim.1
${DOC_} pip install Pyomo
${DOC_} pytest -v --cov=pybnb --cov=examples --cov=src/tests --cov-report="" --run-examples
${DOC_} mv .coverage .coverage.slim.2
