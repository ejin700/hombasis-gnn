"""
Settings for standard treelet stuff.
At some point in the future these kinds of settings files should
be refactored into something nicer.
"""
from settings.globals import *
import os


PATTERN_GRAPHWRAPPER_PARAMS = {'sparse': True}
SPASM_GRAPHWRAPPER_PARAMS = {}

BASEDIR_TUTORIAL = os.path.join(BASEDIR_DATA, 'tutorial')


def raw_pattern_filename(_K=None):
    global BASEDIR_TUTORIAL
    fname = 'patterns.sparse6'
    return os.path.join(BASEDIR_TUTORIAL, fname)


def raw_spasm_filenames(_K=None):
    global BASEDIR_TUTORIAL
    fname = 'spasm_graphs.g6'
    return [os.path.join(BASEDIR_TUTORIAL, fname)]


def pattern_filename_suffix(_K, _suffix=None):
    raise NotImplementedError('Not implemented for the tutorial')


def patterns_filename(_K):
    raise NotImplementedError('Not implemented for the tutorial')


def spasm_filename_suffix(_K, _suffix=None):
    raise NotImplementedError('Not implemented for the tutorial')


def spasm_filename(_K):
    raise NotImplementedError('Not implemented for the tutorial')
