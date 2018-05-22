import os
import sys
import setuptools
import setuptools.command.test
from setuptools import setup, find_packages
from codecs import open

here = os.path.abspath(os.path.dirname(__file__))

about = {}
with open(os.path.join(here, "src", "pybnb", "__about__.py")) as f:
    exec(f.read(), about)

# Get the long description from the README file
def _readme():
    with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
        return f.read()

install_requires = []
with open(os.path.join(here, "requirements.txt")) as f:
    install_requires.extend([ln.strip() for ln in f
                             if ln.strip() != ""])
tests_require = []
with open(os.path.join(here, "test_requirements.txt")) as f:
    tests_require.extend([ln.strip() for ln in f
                          if (ln.strip() != "") and \
                          (not ln.strip() == "-r requirements.txt")])

class PyTest(setuptools.command.test.test):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        super(PyTest, self).initialize_options()
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)

setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__summary__"],
    long_description=_readme(),
    url=about["__uri__"],
    author=about["__author__"],
    author_email=about["__email__"],
    license=about["__license__"],
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords=["optimization","branch and bound"],
    packages=find_packages('src', exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_dir={'':'src'},
    install_requires=install_requires,
    cmdclass={
        "test": PyTest
    },
    # use MANIFEST.in
    include_package_data=True,
    tests_require=tests_require
)
