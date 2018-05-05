#!/bin/bash

set -e

EXAMPLE_ARG=`[[ -z $EXAMPLES ]] || echo --run-examples`
MPIRUN_ARG="mpirun -np 4"`${DOC} mpirun --allow-run-as-root --version 2> /dev/null || echo --allow-run-as-root`
${DOC} mpirun --version 2> /dev/null || echo good
${DOC} pytest -v --doctest-modules src/pybnb
${DOC} pytest -v --cov=pybnb --cov=examples --cov=src/tests --cov-report="" -v ${EXAMPLE_ARG}
${DOC} mv .coverage coverage.parallel.1
${DOC} python run-mpitests.py --mpirun="${MPIRUN_ARG}" --no-build --with-coverage -v
${DOC} mv .coverage coverage.parallel.2
${DOC} python run-mpitests.py --single --no-build --with-coverage -v
${DOC} mv .coverage coverage.parallel.3
# prepare the coverage files for the "coverage combine"
# call that comes next
${DOC} mv coverage.parallel.1 .coverage.parallel.1
${DOC} mv coverage.parallel.2 .coverage.parallel.2
${DOC} mv coverage.parallel.3 .coverage.parallel.3
