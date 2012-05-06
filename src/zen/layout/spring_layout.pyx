#cython: embedsignature=True

"""
Implement spring force layout algorithms.
"""

from zen.graph cimport Graph
from zen.digraph cimport DiGraph
import numpy as np
cimport numpy as np
from zen.exceptions import *
from zen.view import View
cimport cython
from libc.stdlib cimport RAND_MAX, rand

from cpython cimport bool

__all__ = ['layout']

cdef extern from "math.h":
	double sqrt(double x)
	
cdef inline float fast_max(float a, float b): return a if a >= b else b
cdef inline float fast_min(float a, float b): return a if a <= b else b

cpdef layout(GV,bbox):
	"""
	<DOC>
	"""
	view = None
	graph = None

	if type(GV) == View:
		view = GV
		graph = view.graph()
	else:
		graph = GV
		view = View(graph)
		
	raise ZenException, 'Spring layout is not implemented yet'
	
	return view