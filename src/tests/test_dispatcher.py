import pytest

from pybnb.dispatcher import Dispatcher

class TestDispatcherSimple(object):

    def test_no_comm(self):
        with pytest.raises(ValueError):
            Dispatcher(None).serve()
