"""
Settings for standard treelet stuff.
At some point in the future these kinds of settings files should
be refactored into something nicer.
"""
from settings.globals import *
import os


PATTERN_GRAPHWRAPPER_PARAMS = {'sparse': True}
SPASM_GRAPHWRAPPER_PARAMS = {}


def raw_pattern_filename(K):
    global BASEDIR_PATTERNS
    fname = f'patterns_tree{K}.sparse6'
    return os.path.join(BASEDIR_PATTERNS, 'graphs', fname)


def raw_spasm_filenames(K):
    global BASEDIR_SPASMS
    fs = []
    for v in range(2, K + 1):
        fname = f'gen{v}v{K-1}e.g6'
        fs.append(os.path.join(BASEDIR_SPASMS, 'graphs', fname))
    return fs


def pattern_filename_suffix(K, suffix=None):
    global BASEDIR_PATTERNS
    if suffix is None:
        s = f'treelets{K}.dill'
    else:
        s = f'treelets{K}_{suffix}.dill'
    return os.path.join(BASEDIR_PATTERNS, s)


def patterns_filename(K):
    return pattern_filename_suffix(K)


def spasm_filename_suffix(K, suffix=None):
    global BASEDIR_SPASMS
    if suffix is None:
        s = f'treelets{K}_spasm.dill'
    else:
        s = f'treelets{K}_spasm_{suffix}.dill'
    return os.path.join(BASEDIR_SPASMS, s)


def spasm_filename(K):
    return spasm_filename_suffix(K)
