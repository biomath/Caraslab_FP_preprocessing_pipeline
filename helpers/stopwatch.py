from time import time
from numpy import round

# Functions for getting runtime

def tic():
    return time()


def toc(t0, pre_message='Processing time:'):
    tkend = time() - t0
    print(pre_message + ' %d min, %.1f sec' % (tkend // 60, round(tkend % 60, 3)), flush=True)
