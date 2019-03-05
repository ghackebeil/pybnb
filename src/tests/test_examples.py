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

numpy_available = False
try:
    import numpy
    numpy_available = True
except ImportError:
    pass

pyomo_available = False
try:
    import pyomo.kernel as pmo
    if getattr(pmo,"version_info",(0,)*3) >= (5,4,3):  #pragma:nocover
        pyomo_available = True
except:                                                #pragma:nocover
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
            ((not hasattr(ipopt,"executable")) or \
            (ipopt.executable() is not None))

yaml_available = False
try:
    import yaml
    yaml_available = True
except ImportError:
    pass

thisfile = os.path.abspath(__file__)
thisdir = os.path.dirname(thisfile)
topdir = os.path.dirname(
            os.path.dirname(thisdir))
exdir = os.path.join(topdir, "examples")
examples = []
examples.extend(glob.glob(
    os.path.join(exdir,"command_line_problems","*.py")))
examples.extend(glob.glob(
    os.path.join(exdir,"scripts","*.py")))
examples.append(
    os.path.join(exdir,"scripts","tsp","tsp_byvertex.py"))
examples.append(
    os.path.join(exdir,"scripts","tsp","tsp_byedge.py"))
baselinedir = os.path.join(thisdir, "example_baselines")

assert os.path.exists(exdir)
assert thisfile not in examples

tdict = {}
for fname in examples:
    basename = os.path.basename(fname)
    assert basename.endswith(".py")
    assert len(basename) >= 3
    basename = basename[:-3]
    if basename in ("tsp_byvertex",
                    "tsp_byedge"):
        for datafile in ('p01_d',
                         'p01_d_inf'):
            tname = "test_"+basename+"_"+datafile
            bname = os.path.join(baselinedir,
                                 basename+"_"+datafile+".yaml")
            tdict[tname] = (fname,
                            bname,
                            [os.path.join(exdir,"scripts","tsp",
                                          datafile+".txt")])
    else:
        tname = "test_"+basename
        bname = os.path.join(baselinedir,basename+".yaml")
        args = None
        if basename in ("rosenbrock_2d",
                        "lipschitz_1d"):
            args = ["--relative-gap=1e-4"]
        tdict[tname] = (fname,bname,args)
assert len(tdict) == len(examples) + 2

assert "test_binary_knapsack" in tdict
assert len(tdict["test_binary_knapsack"]) == 3
assert "test_binary_knapsack_nested" not in tdict
tdict["test_binary_knapsack_nested"] = \
    (tdict["test_binary_knapsack"][0],
     tdict["test_binary_knapsack"][1],
     ["--nested-solver",
      "--nested-node-limit=100",
      "--nested-queue-strategy=random"])

scenarios = []
for p in [1,2,4]:
    for name in sorted(tdict):
        scenarios.append((name, p))

@pytest.mark.parametrize(("example_name", "procs"),
                         scenarios)
@pytest.mark.example
def test_example(example_name, procs):
    if not yaml_available:
        pytest.skip("yaml is not available")
    if example_name in ("test_bin_packing",
                        "test_rosenbrock_2d",
                        "test_range_reduction_pyomo"):
        if not (pyomo_available and ipopt_available):
            pytest.skip("Pyomo or Ipopt is not available")
    if "tsp_byedge" in example_name:
        if not numpy_available:
            pytest.skip("NumPy is not available")
    if (example_name == "test_simple") or \
       ("tsp_by" in example_name):
        if not mpi4py_available:
            pytest.skip("MPI is not available")
    if (not mpi4py_available) and (procs > 1):
        pytest.skip("MPI is not available")
    filename, baseline_filename, options = tdict[example_name]
    cmd = ["python", filename]
    if options is None:
        options = []
    assert os.path.exists(filename)
    fid, results_filename = tempfile.mkstemp()
    os.close(fid)
    try:
        if procs == 1:
            if ("range_reduction_pyomo" in example_name) or \
               ("tsp_by" in example_name):
                rc = subprocess.call(cmd + \
                                     ["--results-file",
                                      results_filename] + \
                                     options)
            elif example_name == "test_simple":
                rc = subprocess.call(cmd + options)
            else:
                rc = subprocess.call(cmd + \
                                     ["--disable-mpi",
                                      "--results-file",
                                      results_filename] + \
                                     options)
        else:
            assert procs > 1
            if subprocess.call(["mpirun",
                                "--allow-run-as-root",
                                "--version"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT):
                rc = subprocess.call(["mpirun","-np", str(procs)] + \
                                     cmd + \
                                     ["--results-file",
                                      results_filename] + \
                                     options)
            else:
                rc = subprocess.call(["mpirun", "--allow-run-as-root",
                                      "-np", str(procs)] + \
                                     cmd + \
                                     ["--results-file",
                                      results_filename] + \
                                     options)
        assert rc == 0
        if example_name == "test_simple":
            assert not os.path.exists(baseline_filename)
            return
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
