#cython: embedsignature=True

"""
This module provides functions that measure properties of a network.
"""

from graph cimport Graph
from digraph cimport DiGraph
from exceptions import *
from shortest_path cimport *
from constants import *
import numpy as np
cimport numpy as np
import numpy.ma as ma

from cpython cimport bool

__all__ = ['cddist','ddist','diameter','components']

cpdef components(G):
	"""
	Returns a list of the connected components in graph G.  Each component is a set of the node objects in the components.
	"""
	if type(G) == Graph:
		return ug_components(<Graph> G)
	elif type(G) == DiGraph:
		raise InvalidGraphTypeException, 'DiGraph is not yet supported'
		#return dg_components(<DiGraph> G)
	else:
		raise InvalidGraphTypeException, 'Only Graph and DiGraph objects are supported'
	
cpdef ug_components(Graph G):
	all_nodes = set(G.nodes())

	components = []

	while len(all_nodes) > 0:
		seed = all_nodes.pop()
		visited = set()
		to_visit = set([seed])
		while len(to_visit) > 0:
			n = to_visit.pop()
			visited.add(n)
			nbrs = set(G.neighbors(n))
			nbrs.difference_update(visited)
			to_visit.update(nbrs)

		components.append(visited)
		all_nodes.difference_update(visited)

	return components

cpdef dg_components(DiGraph G):
	all_nodes = set(G.nodes())

	components = []

	while len(all_nodes) > 0:
		seed = all_nodes.pop()
		visited = set()
		to_visit = set([seed])
		while len(to_visit) > 0:
			n = to_visit.pop()
			visited.add(n)
			nbrs = set(G.out_neighbors(n))
			nbrs.difference_update(visited)
			to_visit.update(nbrs)

		components.append(visited)
		all_nodes.difference_update(visited)

	return components

cpdef np.ndarray[np.float_t, ndim=1] cddist(G,direction=None,bool inverse=False):
	"""
	Return the cumulative degree distribution of the graph: a numpy
	array, C, where C[i] is the fraction of nodes with degree <= i.
	
	If G is a directed graph and direction is IN_DIR or OUT_DIR, then the
	cumulative distribution will be done with respect to the direction.
	
	If inverse is True, then C[i] is the fraction of nodes with degree >= i.
	"""
	cdef np.ndarray[np.float_t, ndim=1] R = ddist(G,direction,False)
	cdef float seen = 0
	cdef float nnodes = len(G)
	cdef int i
	cdef float x
	
	if inverse:
		seen = nnodes
		for i in range(len(R)):
			x = R[i]
			R[i] = seen / nnodes
			seen -= x
	else:
		for i in range(len(R)):
			seen += R[i]
			R[i] = seen / nnodes
			
	return R

cpdef ddist(G,direction=None,bool normalize=True):
	"""
	Return the degree distribution of the graph - a numpy float array, D, where D[i] is the fraction of nodes with degree i.
	
	If G is a directed graph and direction is IN_DIR or OUT_DIR, then the
	distribution will be done with respect to the direction.
	
	If normalize is False, then D[i] is the number of nodes with degree i.
	"""
	if not G.is_directed() and direction != None:
		raise ZenException, 'Direction cannot be specified for an undirected graph: %s' % direction
		
	if type(G) == Graph:
		return ug_ddist(<Graph> G,normalize)
	elif type(G) == DiGraph:
		return dg_ddist(<DiGraph> G,direction,normalize)
	else:
		raise InvalidGraphTypeException, 'Unknown graph type: %s' % str(type(G))
		
cdef ug_ddist(Graph G,bool normalize):
	cdef int degree,max_degree
	cdef int i
	cdef np.ndarray[np.float_t, ndim=1] dd
	cdef float nnodes
	
	# find the max degree
	max_degree = 0
	for i in range(G.next_node_idx):
		if G.node_info[i].exists:
			degree = G.degree_(i)
			if degree > max_degree:
				max_degree = degree
				
	# compute the degree distribution
	dd = np.zeros( max_degree + 1, np.float )
	
	for i in range(G.next_node_idx):
		if G.node_info[i].exists:
			dd[G.degree_(i)] += 1

	if normalize:
		nnodes = len(G)
		for i in range(max_degree+1):
			dd[i] = dd[i] / nnodes
			
	return dd

cdef dg_ddist(DiGraph G,direction,bool normalize):
	cdef int degree,max_degree
	cdef int i
	cdef np.ndarray[np.float_t, ndim=1] dd
	cdef float nnodes
	
	if direction is None or direction == BOTH_DIR:
		# find the max degree
		max_degree = 0
		for i in range(G.next_node_idx):
			if G.node_info[i].exists:
				degree = G.degree_(i)
				if degree > max_degree:
					max_degree = degree

		# compute the degree distribution
		dd = np.zeros( max_degree + 1, np.float )

		for i in range(G.next_node_idx):
			if G.node_info[i].exists:
				dd[G.degree_(i)] += 1

		if normalize:
			nnodes = len(G)
			for i in range(max_degree+1):
				dd[i] = dd[i] / nnodes

		return dd
	elif direction == IN_DIR:
		# find the max degree
		max_degree = 0
		for i in range(G.next_node_idx):
			if G.node_info[i].exists:
				degree = G.in_degree_(i)
				if degree > max_degree:
					max_degree = degree

		# compute the degree distribution
		dd = np.zeros( max_degree + 1, np.float )

		for i in range(G.next_node_idx):
			if G.node_info[i].exists:
				dd[G.in_degree_(i)] += 1

		if normalize:
			nnodes = len(G)
			for i in range(max_degree+1):
				dd[i] = dd[i] / nnodes

		return dd
	elif direction == OUT_DIR:
		# find the max degree
		max_degree = 0
		for i in range(G.next_node_idx):
			if G.node_info[i].exists:
				degree = G.out_degree_(i)
				if degree > max_degree:
					max_degree = degree

		# compute the degree distribution
		dd = np.zeros( max_degree + 1, np.float )

		for i in range(G.next_node_idx):
			if G.node_info[i].exists:
				dd[G.out_degree_(i)] += 1

		if normalize:
			nnodes = len(G)
			for i in range(max_degree+1):
				dd[i] = dd[i] / nnodes

		return dd
	else:
		raise ZenException, 'Invalid direction: %s' % direction

cpdef diameter(G):
	"""
	Return the diameter of the graph - the longest shortest path in the graph.
	"""
	P = floyd_warshall_path_length_(G)
	
	return <int> ma.masked_equal(P,float('infinity')).max()