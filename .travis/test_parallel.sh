#!/bin/bash

set -e

EXAMPLE_ARG=`[[ -z $EXAMPLES ]] || echo --run-examples`
MPIRUN_FLAG="--mpirun="
if [ -z "$MPIRUN_ARG" ]
then
    if [ "$(${DOC} mpirun --allow-run-as-root --version 2> /dev/null)" ]
    then
        MPIRUN_ARG="mpirun --allow-run-as-root -n 4"
    else
        MPIRUN_ARG="mpirun -n 4"
    fi
fi
[[ -n $NODOCTEST ]] || ${DOC} pytest -v --doctest-plus src/pybnb
${DOC} pytest -v --cov=pybnb --cov=examples --cov=src/tests --cov-report="" -v ${EXAMPLE_ARG}
${DOC} mv .coverage coverage.parallel.1
${DOC} python run-mpitests.py "${MPIRUN_FLAG}""${MPIRUN_ARG}" --no-build --with-coverage -v
${DOC} mv .coverage coverage.parallel.2
# prepare the coverage files for the "coverage combine"
# call that comes next
${DOC} mv coverage.parallel.1 .coverage.parallel.1
${DOC} mv coverage.parallel.2 .coverage.parallel.2
