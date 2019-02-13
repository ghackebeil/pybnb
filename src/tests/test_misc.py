import os
import tempfile
import logging
import signal

import pytest

from pybnb.common import (inf,
                          nan)
from pybnb.misc import (_cast_to_float_or_int,
                        MPI_InterruptHandler,
                        metric_format,
                        time_format,
                        get_gap_labels,
                        as_stream,
                        get_default_args,
                        get_keyword_docs,
                        get_simple_logger)

from six import StringIO

yaml_available = False
try:
    import yaml
    yaml_available = True
except ImportError:
    pass

numpy_available = False
try:
    import numpy
    numpy_available = True
except ImportError:
    pass

class Test(object):

    def test_MPI_InterruptHandler(self):
        assert len(MPI_InterruptHandler._sigs) > 0
        original_handlers = \
            [(signum, signal.getsignal(signum))
             for signum in MPI_InterruptHandler._sigs]
        with MPI_InterruptHandler(lambda s,f: None) as h:
            assert not h._released
        assert h._released
        for i, signum in enumerate(
                MPI_InterruptHandler._sigs):
            orig = signal.getsignal(signum)
            assert original_handlers[i][0] == signum
            assert original_handlers[i][1] is orig

        def fn(s,f):
            fn.called = True
        fn.called = False
        with MPI_InterruptHandler(fn) as h:
            assert not h._released
            signum = MPI_InterruptHandler._sigs[0]
            signal.getsignal(signum)(None, None)
            assert h._released
            for i, signum in enumerate(
                MPI_InterruptHandler._sigs):
                orig = signal.getsignal(signum)
                assert original_handlers[i][0] == signum
                assert original_handlers[i][1] is orig
        assert fn.called

    def test_metric_format(self):
        assert metric_format(None) == "<unknown>"
        assert metric_format(0.0) == "0.0 s"
        assert metric_format(0.0, align_unit=True) == "0.0 s "
        assert metric_format(0.0, unit='B') == "0.0 B"
        assert metric_format(0.0, digits=2) == "0.00 s"
        assert metric_format(1000.23, digits=3) == "1.000 Ks"
        assert metric_format(1000.23, digits=4) == "1.0002 Ks"
        assert metric_format(1000000.23, digits=4) == "1.0000 Ms"
        assert metric_format(0.23334, digits=1) == "233.3 ms"
        assert metric_format(0.23334, digits=2) == "233.34 ms"
        assert metric_format(0.00023334, digits=1) == "233.3 us"
        assert metric_format(0.00023334, digits=2) == "233.34 us"

    def test_time_format(self):
        assert time_format(None) == "<unknown>"
        assert time_format(0.0) == "0.0 s"
        assert time_format(0.0, align_unit=True) == "0.0 s "
        assert time_format(0.0, digits=2) == "0.00 s"
        assert time_format(24.9) == "24.9 s"
        assert time_format(93.462, digits=3) == "1.558 m"
        assert time_format(93.462, digits=4) == "1.5577 m"
        assert time_format(93.462, digits=4,
                           align_unit=True) == "1.5577 m "
        assert time_format(5607.72, digits=3) == "1.558 h"
        assert time_format(5607.72, digits=4) == "1.5577 h"
        assert time_format(5607.72, digits=4,
                           align_unit=True) == "1.5577 h "
        assert time_format(134585.28, digits=3) == "1.558 d"
        assert time_format(134585.28, digits=4) == "1.5577 d"
        assert time_format(134585.28, digits=4,
                           align_unit=True) == "1.5577 d "
        assert time_format(0.23334, digits=1) == "233.3 ms"
        assert time_format(0.23334, digits=2) == "233.34 ms"

    def test_get_gap_labels(self):
        l0, l1, l2 = get_gap_labels(1)
        assert l0 == 10
        assert l1 == "{gap:>10}"
        assert l2 == "{gap:>10.1f}"
        l0, l1, l2 = get_gap_labels(0.1)
        assert l0 == 10
        assert l1 == "{gap:>10}"
        assert l2 == "{gap:>10.2f}"
        l0, l1, l2 = get_gap_labels(0.01)
        assert l0 == 10
        assert l1 == "{gap:>10}"
        assert l2 == "{gap:>10.3f}"
        l0, l1, l2 = get_gap_labels(0.001)
        assert l0 == 10
        assert l1 == "{gap:>10}"
        assert l2 == "{gap:>10.4f}"
        l0, l1, l2 = get_gap_labels(0.0001)
        assert l0 == 10
        assert l1 == "{gap:>10}"
        assert l2 == "{gap:>10.5f}"
        l0, l1, l2 = get_gap_labels(0.00001)
        assert l0 == 11
        assert l1 == "{gap:>11}"
        assert l2 == "{gap:>11.6f}"
        l0, l1, l2 = get_gap_labels(0.000001,key='rgap')
        assert l0 == 12
        assert l1 == "{rgap:>12}"
        assert l2 == "{rgap:>12.7f}"
        l0, l1, l2 = get_gap_labels(0.0000001,key='agap',format='g')
        assert l0 == 13
        assert l1 == "{agap:>13}"
        assert l2 == "{agap:>13.8g}"

    def test_as_stream(self):
        fid, fname = tempfile.mkstemp()
        os.close(fid)
        with as_stream(fname) as f:
            assert not f.closed
            assert hasattr(f,'write')
        assert f.closed
        fid, fname = tempfile.mkstemp()
        os.close(fid)
        with as_stream(u""+fname) as f:
            assert not f.closed
            assert hasattr(f,'write')
        assert f.closed
        with open(fname) as f:
            assert not f.closed
            with as_stream(f) as f_:
                assert f is f_
                assert not f.closed
            assert not f.closed

    def test_get_default_args(self):
        def f(a):                                 #pragma:nocover
            pass
        assert get_default_args(f) == {}
        def f(a, b):                              #pragma:nocover
            pass
        assert get_default_args(f) == {}
        def f(*args):                             #pragma:nocover
            pass
        assert get_default_args(f) == {}
        def f(**kwds):                            #pragma:nocover
            pass
        assert get_default_args(f) == {}
        def f(*args, **kwds):                     #pragma:nocover
            pass
        assert get_default_args(f) == {}
        def f(a, b=1):                            #pragma:nocover
            pass
        assert get_default_args(f) == {'b':1}
        def f(a=1):                               #pragma:nocover
            pass
        assert get_default_args(f) == {'a':1}
        def f(a=(1,)):                            #pragma:nocover
            pass
        assert get_default_args(f) == {'a':(1,)}

    def test_get_keyword_docs(self):
        if not yaml_available:
            pytest.skip("yaml is not available")
        import pybnb.solver
        data = get_keyword_docs(pybnb.solver.Solver.solve.__doc__)
        kwds = get_default_args(pybnb.solver.Solver.solve)
        assert len(data) > 1
        for key in data:
            if 'default' in data[key]:
                assert data[key]['default'] == kwds[key]
            assert "choices" not in data[key]
        def f():
            """Something

            Parameters
            ----------
            junk1 : {"a", "b", 1}
                Junk1 description.
            junk2 : {"c", "d"}, optional
                Junk2 description more than one
                line. (default: "c")
            junk3 : int
                Junk3 description.
            """
        data = get_keyword_docs(f.__doc__)
        assert data == \
            {'junk1': {'choices': ['a', 'b', 1],
                       'doc': 'Junk1 description.'},
             'junk2': {'choices': ['c', 'd'],
                       'default': 'c',
                       'doc': 'Junk2 description more than one line.'},
             'junk3': {'doc': 'Junk3 description.'}}

    def test_get_simple_logger(self):
        log = get_simple_logger(console=False)
        assert log.disabled
        log = get_simple_logger()
        assert not log.disabled
        log = get_simple_logger(console=True)
        assert not log.disabled
        assert len(log.handlers) == 2
        log.info('junk')
        fid, fname = tempfile.mkstemp()
        out = StringIO()
        os.close(fid)
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        try:
            log = get_simple_logger(filename=fname,
                                    stream=out,
                                    console=True,
                                    formatter=formatter,
                                    level=logging.WARNING)
            assert len(log.handlers) == 4
            log.error('error_line')
            log.warning('warning_line')
            log.info('info_line')
            log.debug('debug_line')
            for handler in log.handlers:
                handler.close()
            with open(fname) as f:
                lines = f.readlines()
                assert len(lines) == 2
                assert lines[0].strip() == '[ERROR] error_line'
                assert lines[1].strip() == '[WARNING] warning_line'
                del lines
            lines = out.getvalue().splitlines()
            assert lines[0].strip() == '[ERROR] error_line'
            assert lines[1].strip() == '[WARNING] warning_line'
        finally:
            os.remove(fname)

    def test_cast_to_float_or_int(self):
        assert type(_cast_to_float_or_int(inf)) is float
        assert type(_cast_to_float_or_int(nan)) is float
        assert type(_cast_to_float_or_int(1.0)) is float
        assert type(_cast_to_float_or_int(1.1)) is float
        assert type(_cast_to_float_or_int(1)) is int
        assert type(_cast_to_float_or_int(True)) is int
        with pytest.raises(TypeError):
            _cast_to_float_or_int(None)
        if numpy_available:
            numpy_types = []
            numpy_types.append(('bool', int))
            numpy_types.append(('bool_', float)) # edge case
            numpy_types.append(('int_', int))
            numpy_types.append(('intc', int))
            numpy_types.append(('intp', int))
            numpy_types.append(('int8', int))
            numpy_types.append(('int16', int))
            numpy_types.append(('int32', int))
            numpy_types.append(('int64', int))
            numpy_types.append(('uint8', int))
            numpy_types.append(('uint16', int))
            numpy_types.append(('uint32', int))
            numpy_types.append(('uint64', int))
            numpy_types.append(('float_', float))
            numpy_types.append(('float16', float))
            numpy_types.append(('float32', float))
            numpy_types.append(('float64', float))
            numpy_types.append(('float128', float))
            numpy_types.append(('complex_', float))
            numpy_types.append(('complex64', float))
            numpy_types.append(('complex128', float))
            for name, cast_type in numpy_types:
                try:
                    type_ = getattr(numpy, name)
                except:                           #pragma:nocover
                    continue
                assert type(_cast_to_float_or_int(
                    type_())) is cast_type
