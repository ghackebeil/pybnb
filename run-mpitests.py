import sys
from runtests.mpi import Tester
import os.path
# code is placed inside this block so that it is not
# executed when imported during doctests
if __name__ == "__main__":
    tester = Tester(os.path.join(os.path.abspath(__file__)),
                "../../src", # tests are executed from within ./build/tests
                extra_path=None)
    tester.main(sys.argv[1:])
