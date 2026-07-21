"""Minimal stopwatch utility for timing and reporting the duration of pipeline stages."""
from time import time
from numpy import round


def tic():
    """Start a stopwatch.

    Returns
    -------
    float
        Current time in seconds since the epoch, as returned by ``time.time()``.
        Pass this value to :func:`toc` to compute an elapsed duration.
    """
    return time()


def toc(t0, pre_message='Processing time:'):
    """Stop a stopwatch and print the elapsed time since ``t0``.

    Parameters
    ----------
    t0 : float
        Start time as returned by :func:`tic`.
    pre_message : str, optional
        Text to print before the elapsed-time string. Default is ``'Processing time:'``.

    Returns
    -------
    None
    """
    tkend = time() - t0
    print(pre_message + ' %d min, %.1f sec' % (tkend // 60, round(tkend % 60, 3)), flush=True)
