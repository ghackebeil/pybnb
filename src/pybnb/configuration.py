"""
Configuration settings for node serialization.

Copyright by Gabriel A. Hackebeil (gabe.hackebeil@gmail.com).
"""
import os
import platform
import pickle

from pybnb import __version__

class Configuration(object):
    """The main configuration object.

    Attributes
    ----------
    SERIALIZER : str, {'pickle', 'dill'}
        The name of serialization module used to transmit
        node state. (default: "pickle")
    SERIALIZER_PROTOCOL_VERSION : int
        The protocol argument passed to the `dumps` function
        of the selected serialization module.
        (default: pickle.HIGHEST_PROTOCOL)
    COMPRESSION : bool
        Indicates whether or not to compress the serialized
        node state using zlib. (default: False)
    MARSHAL_PROTOCOL_VERSION : int
        The version argument passed to the
        :func:`marshal.dumps` function. (default: 2)
    """
    __slots__ = ("SERIALIZER",
                 "SERIALIZER_PROTOCOL_VERSION",
                 "COMPRESSION",
                 "MARSHAL_PROTOCOL_VERSION")

    def __init__(self):
        self.reset()

    def reset(self, use_environment=True):
        """Reset the configuration to default settings.

        Parameters
        ----------
        use_environment : bool, optional
            Controls whether or not to check for environment
            variables to overwrite the default
            settings. (default: True)
        """
        self.SERIALIZER = "pickle"
        self.SERIALIZER_PROTOCOL_VERSION = pickle.HIGHEST_PROTOCOL
        self.COMPRESSION = False
        self.MARSHAL_PROTOCOL_VERSION = 2
        if use_environment:
            # process environment variables
            prefix = "PYBNB_"
            for symbol in self.__slots__:
                if prefix+symbol in os.environ:
                    default = getattr(self, symbol)
                    value = os.environ[prefix+symbol]
                    if symbol == "COMPRESSION":
                        if value in ("0",
                                     "off","Off","OFF",
                                     "no","No","NO",
                                     "false","False","FALSE"):
                            value = False
                        elif value in ("1",
                                       "on","On","ON",
                                       "yes","Yes","YES",
                                       "true","True","TRUE"):
                            value = True
                        else:
                            raise ValueError(
                                "invalid boolean value: %s%s=%s"
                                % (prefix, symbol, value))
                    else:
                        value = type(default)(value)
                    setattr(self, symbol, value)

    def __str__(self):
        out =  "pybnb version: %s\n" % __version__
        out += ("loaded from: %s\n"
                % (os.path.dirname(__file__)))
        out += ("python version: %s %s (%s, %s)\n"
                % (platform.python_implementation(),
                   platform.python_version(),
                   platform.system(),
                   os.name))
        out += "configuration:"
        for key in self.__slots__:
            out += ("\n - %s: %s" % (key,
                                     getattr(self, key)))
        return out

config = Configuration()

if __name__ == "__main__":                        #pragma:nocover
    print(config)
