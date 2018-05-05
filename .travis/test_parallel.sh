#!/bin/bash

set -e

EXAMPLE_ARG=`[[ -z $EXAMPLES ]] || echo --run-examples`
if [ -z "$MPIRUN_ARG" ]
then
    if [ "$(${DOC} mpirun --allow-run-as-roots --version 2> /dev/null)" ]
    then
        MPIRUN_ARG="--mpirun \"mpirun --allow-run-as-root np 4\""
    else
        MPIRUN_ARG="--mpirun \"mpirun -np 4\""
    fi
fi
echo ${MPIRUN_ARG}
${DOC} pytest -v --doctest-modules src/pybnb
${DOC} pytest -v --cov=pybnb --cov=examples --cov=src/tests --cov-report="" -v ${EXAMPLE_ARG}
${DOC} mv .coverage coverage.parallel.1
${DOC} python run-mpitests.py ${MPIRUN_ARG} --no-build --with-coverage -v
${DOC} mv .coverage coverage.parallel.2
${DOC} python run-mpitests.py --single --no-build --with-coverage -v
${DOC} mv .coverage coverage.parallel.3
# prepare the coverage files for the "coverage combine"
# call that comes next
${DOC} mv coverage.parallel.1 .coverage.parallel.1
${DOC} mv coverage.parallel.2 .coverage.parallel.2
${DOC} mv coverage.parallel.3 .coverage.parallel.3
