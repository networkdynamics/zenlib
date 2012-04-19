"""
This module provides functions that measure of distances within a network.
"""

from shortest_path cimport *
import numpy as np
cimport numpy as np
import numpy.ma as ma

__all__ = ['diameter']

cpdef diameter(G):
	"""
	Return the diameter of the graph - the longest shortest path in the graph.
	"""
	P = floyd_warshall_path_length_(G)
	
	return <int> ma.masked_equal(P,float('infinity')).max()