#!/bin/bash

set -e

EXAMPLE_ARG=`[[ -z $EXAMPLES ]] || echo --run-examples`
${DOC_} pytest -v --doctest-modules src/pybnb
${DOC_} pytest -v --cov=pybnb --cov=examples --cov=src/tests --cov-report="" -v ${EXAMPLE_ARG}
${DOC_} mv .coverage .coverage.parallel.1
${DOC_} python run-mpitests.py --no-build --with-coverage -v
${DOC_} mv .coverage .coverage.parallel.2
${DOC_} python run-mpitests.py --single --no-build --with-coverage -v
${DOC_} mv .coverage .coverage.parallel.3
