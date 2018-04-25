# Configure default usage of pytest in this package
import pytest

def pytest_addoption(parser):
    parser.addoption("--run-examples",
                     action="store_true",
                     default=False,
                     help=("run tests on examples "
                           "(this disabled other mpi tests)"))

def pytest_ignore_collect(path, config):
    if config.getoption("--run-examples"):
        if "src/tests/mpi" in str(path):
            return True

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-examples"):
        skip_example = pytest.mark.skip(reason="need --run-examples option to run")
        for item in items:
            if "example" in item.keywords:
                item.add_marker(skip_example)
