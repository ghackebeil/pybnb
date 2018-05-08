import os
import tempfile
import logging

from pybnb.misc import (metric_fmt,
                        get_gap_labels,
                        as_stream,
                        get_default_args,
                        get_keyword_docs,
                        get_simple_logger)

from six import StringIO

import numpy

class Test(object):

    def test_metric_fmt(self):
        assert metric_fmt(None) == "<unknown>"
        assert metric_fmt(0.0) == "0.0 s"
        assert metric_fmt(0.0,unit='B') == "0.0 B"
        assert metric_fmt(0.0, digits=2) == "0.00 s"
        assert metric_fmt(1000.23, digits=3) == "1.000 Ks"
        assert metric_fmt(1000.23, digits=4) == "1.0002 Ks"
        assert metric_fmt(1000000.23, digits=4) == "1.0000 Ms"
        assert metric_fmt(0.23334, digits=1) == "233.3 ms"
        assert metric_fmt(0.23334, digits=2) == "233.34 ms"

    def test_get_gap_labels(self):
        l1, l2 = get_gap_labels(1)
        assert l1 == "{gap:>10}"
        assert l2 == "{gap:>10.1f}"
        l1, l2 = get_gap_labels(0.1)
        assert l1 == "{gap:>10}"
        assert l2 == "{gap:>10.2f}"
        l1, l2 = get_gap_labels(0.01)
        assert l1 == "{gap:>10}"
        assert l2 == "{gap:>10.3f}"
        l1, l2 = get_gap_labels(0.001)
        assert l1 == "{gap:>10}"
        assert l2 == "{gap:>10.4f}"
        l1, l2 = get_gap_labels(0.0001)
        assert l1 == "{gap:>10}"
        assert l2 == "{gap:>10.5f}"
        l1, l2 = get_gap_labels(0.00001)
        assert l1 == "{gap:>11}"
        assert l2 == "{gap:>11.6f}"
        l1, l2 = get_gap_labels(0.000001,key='rgap')
        assert l1 == "{rgap:>12}"
        assert l2 == "{rgap:>12.7f}"
        l1, l2 = get_gap_labels(0.0000001,key='agap',format='g')
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
        def f(a=[]):
            a.append(1)
        assert get_default_args(f) == {'a':[]}
        f()
        assert get_default_args(f) == {'a':[1]}

    def test_get_keyword_docs(self):
        import pybnb.solver
        data = get_keyword_docs(pybnb.solver.Solver.solve.__doc__)
        kwds = get_default_args(pybnb.solver.Solver.solve)
        for key in data:
            if 'default' in data[key]:
                assert data[key]['default'] == kwds[key]

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
