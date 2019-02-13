import os
import pickle

from pybnb import config as _config_
from pybnb.configuration import Configuration

import pytest

class TestConfiguration(object):

    def test_str(self):
        print(_config_)

    def test_reset(self):
        config = Configuration()
        config.reset(use_environment=False)
        assert config.SERIALIZER == "pickle"
        assert config.SERIALIZER_PROTOCOL_VERSION == \
            pickle.HIGHEST_PROTOCOL
        assert not config.COMPRESSION
        assert config.MARSHAL_PROTOCOL_VERSION == 2
        env_orig = {}
        prefix = "PYBNB_"
        for symbol in Configuration.__slots__:
            key = prefix+symbol
            if key in os.environ:
                env_orig[key] = os.environ[key]
        try:
            os.environ["PYBNB_SERIALIZER"] = "dill"
            os.environ["PYBNB_SERIALIZER_PROTOCOL_VERSION"] = "0"
            os.environ["PYBNB_COMPRESSION"] = "1"
            os.environ["PYBNB_MARSHAL_PROTOCOL_VERSION"] = "0"
            config.reset(use_environment=False)
            assert config.SERIALIZER == "pickle"
            assert config.SERIALIZER_PROTOCOL_VERSION == \
                pickle.HIGHEST_PROTOCOL
            assert not config.COMPRESSION
            assert config.MARSHAL_PROTOCOL_VERSION == 2
            config.reset(use_environment=True)
            assert config.SERIALIZER == "dill"
            assert config.SERIALIZER_PROTOCOL_VERSION == 0
            assert config.COMPRESSION
            assert config.MARSHAL_PROTOCOL_VERSION == 0
            os.environ["PYBNB_COMPRESSION"] = "0"
            config.reset(use_environment=True)
            assert not config.COMPRESSION
            os.environ["PYBNB_COMPRESSION"] = "_not_a_bool_"
            with pytest.raises(ValueError):
                config.reset(use_environment=True)
        finally:
            # reset the environment to its original state
            for symbol in Configuration.__slots__:
                key = prefix+symbol
                if key in env_orig:
                    os.environ[key] = env_orig[key]
                else:
                    del os.environ[key]
