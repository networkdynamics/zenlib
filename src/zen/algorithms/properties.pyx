"""
This module provides functions that measure of distances within a network.
"""

from shortest_path cimport *
import numpy as np
cimport numpy as np
import numpy.ma as ma

__all__ = ['diameter']

cpdef diameter(G,ignore_weights=True):
	"""
	Return the diameter of the graph - the longest shortest path in the graph.

	**Args**:

		* ``ignore_weights [=False]`` (boolean): whether edge weights should influence the diameter.

	"""
	if ignore_weights:
		P = floyd_warshall_path_length_(G)
	else:
		P = all_pairs_dijkstra_path_length_(G,ignore_weights=False)

	return <float> ma.masked_equal(P,float('infinity')).max()
