import os
import glob
import subprocess
import tempfile

import pytest

mpi4py_available = False
try:
    import mpi4py
    mpi4py_available = True
except ImportError:
    pass

pyomo_available = False
try:
    import pyomo.kernel as pmo
    if getattr(pmo,'version_info',(0,)*3) >= (5,4,3):  #pragma:nocover
        pyomo_available = True
except ImportError:                                    #pragma:nocover
    pass

ipopt_available = False
if pyomo_available:                                    #pragma:nocover
    from pyomo.opt.base import UnknownSolver
    ipopt = pmo.SolverFactory("ipopt")
    if ipopt is None or isinstance(ipopt, UnknownSolver):
        ipopt_available = False
    else:
        ipopt_available = \
            (ipopt.available(exception_flag=False)) and \
            ((not hasattr(ipopt,'executable')) or \
            (ipopt.executable() is not None))

import yaml

thisfile = os.path.abspath(__file__)
thisdir = os.path.dirname(thisfile)
topdir = os.path.dirname(
            os.path.dirname(thisdir))
exdir = os.path.join(topdir, "examples")
examples = glob.glob(os.path.join(exdir,"*.py"))
baselinedir = os.path.join(thisdir, "example_baselines")

assert os.path.exists(exdir)
assert thisfile not in examples

tdict = {}
for fname in examples:
    basename = os.path.basename(fname)
    assert basename.endswith(".py")
    assert len(basename) >= 3
    basename = basename[:-3]
    tname = "test_"+basename
    bname = os.path.join(baselinedir,basename+".yaml")
    tdict[tname] = (fname,bname)
assert len(tdict) == len(examples)

scenarios = []
for p in [1,2,4]:
    for name in sorted(tdict):
        scenarios.append((name, p))

@pytest.mark.parametrize(("example_name", "procs"),
                         scenarios)
@pytest.mark.example
def test_example(example_name, procs):
    if example_name in ("test_bin_packing",
                        "test_rosenbrock_2d",
                        "test_range_reduction_script"):
        if not (pyomo_available and ipopt_available):
            pytest.skip("Pyomo or Ipopt is not available")
    if (not mpi4py_available) and (procs > 1):
            pytest.skip("MPI is not available")
    filename, baseline_filename = tdict[example_name]
    assert os.path.exists(filename)
    fid, results_filename = tempfile.mkstemp()
    os.close(fid)
    try:
        if procs == 1:
            if example_name == "test_range_reduction_script":
                rc = subprocess.call(['python', filename,
                                      "--results-file", results_filename])
            else:
                rc = subprocess.call(['python', filename, '--disable-mpi',
                                      "--results-file", results_filename])
        else:
            assert procs > 1
            if subprocess.call(["mpirun",
                                "--allow-run-as-root",
                                "--version"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT):
                rc = subprocess.call(["mpirun",
                                      "-np", str(procs),
                                      'python', filename,
                                      "--results-file", results_filename])
            else:
                rc = subprocess.call(["mpirun",
                                      "--allow-run-as-root",
                                      "-np", str(procs),
                                      'python', filename,
                                      "--results-file", results_filename])
        assert rc == 0
        with open(results_filename) as f:
            results = yaml.load(f)
        with open(baseline_filename) as f:
            baseline_results = yaml.load(f)
        assert len(baseline_results) < len(results)
        for key in baseline_results:
            if type(baseline_results[key]) is float:
                assert round(baseline_results[key], 4) == \
                    round(results[key], 4)
            else:
                assert baseline_results[key] == results[key]
    finally:
        os.remove(results_filename)
