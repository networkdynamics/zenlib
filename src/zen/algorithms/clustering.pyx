"""
The ``zen.algorithms.clustering`` module (available as ``zen.clustering``) implements three measures of the degree of local clustering of nodes
in a network's connectivity:

	* `Global clustering coefficient <http://en.wikipedia.org/wiki/Clustering_coefficient#Global_clustering_coefficient>`_
	* `Local clustering coefficient <http://en.wikipedia.org/wiki/Clustering_coefficient#Local_clustering_coefficient>`_
	* `Network average clustering coefficient <http://en.wikipedia.org/wiki/Clustering_coefficient#Network_average_clustering_coefficient>`_

Functions
---------

.. autofunction:: gcc(G)

.. autofunction:: ncc(G)

.. autofunction:: lcc(G)

.. autofunction:: lcc_(G)

"""

from zen.digraph cimport DiGraph
from zen.graph cimport Graph
import numpy as np
cimport numpy as np
from exceptions import *

from cpython cimport bool

__all__ = ['gcc','lcc','lcc_','ncc']
	
cpdef float gcc(G):
	"""
	Compute the `global clustering coefficient <http://en.wikipedia.org/wiki/Clustering_coefficient#Global_clustering_coefficient>`_: the fraction of triangles to vee's in the network.
	"""
	if type(G) == DiGraph:
		return __gcc_directed(<DiGraph> G)
	elif type(G) == Graph:
		return __gcc_undirected(<Graph> G)
	else:
		raise ZenException, 'Graph type %s not supported' % str(type(G))

cpdef float __gcc_directed(DiGraph G):
	cdef int i,j,k
	cdef int ni,nj,nk,nl
	cdef int num_vees = 0
	cdef int num_tris = 0
	cdef int num_notself

	for ni in range(G.next_node_idx):

		if not G.node_info[ni].exists:
			continue

		num_notself = 0

		# check each edge
		for i in range(G.node_info[ni].outdegree):
			nj = G.edge_info[G.node_info[ni].outelist[i]].tgt

			if nj == ni:
				continue
			else:
				num_notself += 1

			for j in range(G.node_info[nj].outdegree):
				nk = G.edge_info[G.node_info[nj].outelist[j]].tgt
				for k in range(G.node_info[nk].indegree):
					nl = G.edge_info[G.node_info[nk].inelist[k]].src
					if nl == ni:
						num_tris += 1

		num_vees += num_notself * (num_notself - 1)

	if num_vees == 0:
		return 0
	else:
		return <float> num_tris / <float> num_vees

cpdef float __gcc_undirected(Graph G):
	cdef int idx
	cdef int i,j,k
	cdef int ni,nj,nk,nl
	cdef int num_vees = 0
	cdef int num_tris = 0
	cdef int num_nonself = 0

	# count the number of unique triangles (regardless of node ordering)
	for ni in range(G.next_node_idx):

		if not G.node_info[ni].exists:
			continue

		# loop over all nodes adjacent to ni
		num_nonself = 0
		for i in range(G.node_info[ni].degree):
			nj = G.endpoint_(G.node_info[ni].elist[i],ni)

			# keep track of how many of ni's edges are non-self loops.  This 
			# is used to compute the number of V's
			if nj != ni:
				num_nonself += 1

			if nj <= ni:
				continue

			# loop over all nodes adjacent to nj
			for j in range(G.node_info[nj].degree):
				nk = G.endpoint_(G.node_info[nj].elist[j],nj)

				if nk <= nj:
					continue

				for k in range(G.node_info[nk].degree):
					nl = G.endpoint_(G.node_info[nk].elist[k],nk)
					if nl == ni:
						num_tris += 1

		num_vees += (num_nonself * (num_nonself - 1)) / 2

	if num_vees == 0:
		return 0
	else:
		return <float> (3 * num_tris) / <float> num_vees

cpdef lcc(G):
	"""
	Compute the `local clustering coefficients <http://en.wikipedia.org/wiki/Clustering_coefficient#Local_clustering_coefficient>`_ for all nodes in the graph, ``G``.
	
	**Returns**:
		:py:class:`dict`, ``C``.  ``C[n]`` is the local clustering coefficient for the node identified by node object ``n``.
	"""
	cdef np.ndarray[np.float_t, ndim=1] C = lcc_(G)
	cdef int i
	
	R = dict()
	for i in range(len(C)):
		if C[i] >= 0:
			R[G.node_object(i)] = C[i]
	
	return R
				
cpdef np.ndarray[np.float_t, ndim=1] lcc_(G):
	"""
	Compute the `local clustering coefficients <http://en.wikipedia.org/wiki/Clustering_coefficient#Local_clustering_coefficient>`_ for all nodes in the graph, ``G``.
	
	**Returns**:
		1D ``numpy.ndarray``, ``C``.  ``C[i]`` is the local clustering coefficient for the node with
		index ``i``.
	"""
	if type(G) == DiGraph:
		return __lcc_directed(<DiGraph>G)
	elif type(G) == Graph:
		return __lcc_undirected(<Graph>G)
	else:
		raise ZenException, 'Unsupported graph type: %s' % str(type(G))

cpdef np.ndarray[np.float_t, ndim=1] __lcc_directed(DiGraph G):
	"""
	Compute the local clustering coefficient for nodes in the directed graph G.
	"""
	cdef np.ndarray[np.float_t, ndim=1] C = np.empty(G.next_node_idx, np.float)
	C.fill(-1.0)
	
	cdef int num_nodes = 0
	cdef int i,j,k
	cdef int ni,nj,nk
	cdef int num_vees = 0
	cdef int num_tris = 0

	for ni in range(G.next_node_idx):
		if not G.node_info[ni].exists:
			continue
		
		num_vees = 0
		num_tris = 0
		
		# check each edge
		for i in range(G.node_info[ni].outdegree):
			nj = G.edge_info[G.node_info[ni].outelist[i]].tgt
			
			if nj == ni:
				continue
				
			for j in range(G.node_info[ni].outdegree):				
				nk = G.edge_info[G.node_info[ni].outelist[j]].tgt
				
				if nk == ni or nk == nj:
					continue
					
				num_vees += 1
				if G.has_edge_(nj,nk):
					num_tris += 1

		# compute the icc for this node
		if num_vees == 0:
			C[ni] = 0.0
		else:
			C[ni] = <float> num_tris / <float> num_vees

	return C
		
cpdef np.ndarray[np.float_t, ndim=1] __lcc_undirected(Graph G):
	"""
	Compute the local clustering coefficient for nodes in the directed graph G.
	"""
	cdef np.ndarray[np.float_t, ndim=1] C = np.empty(G.next_node_idx, np.float)
	C.fill(-1.0)
	
	cdef int i,j,k
	cdef int ni,nj,nk
	cdef int num_vees = 0
	cdef int num_tris = 0

	for ni in range(G.next_node_idx):
		if not G.node_info[ni].exists:
			continue

		num_vees = 0
		num_tris = 0

		# check each edge
		for i in range(G.node_info[ni].degree):
			if G.edge_info[G.node_info[ni].elist[i]].u == ni:
				nj = G.edge_info[G.node_info[ni].elist[i]].v
			else:
				nj = G.edge_info[G.node_info[ni].elist[i]].u
			
			if ni == nj:
				continue
				
			for j in range(G.node_info[ni].degree):
				if G.edge_info[G.node_info[ni].elist[j]].u == ni:
					nk = G.edge_info[G.node_info[ni].elist[j]].v
				else:
					nk = G.edge_info[G.node_info[ni].elist[j]].u
				
				if nk == ni or nk == nj:
					continue
					
				num_vees += 1
				
				if G.has_edge_(nj,nk):
					num_tris += 1
				
		if num_vees == 0:
			C[ni] = 0.0
		else:
			C[ni] = <float> num_tris / <float> num_vees
			
		idx += 1

	return C
	
cpdef float ncc(G) except -1:
	"""
	Compute the `network (average) clustering coefficient <http://en.wikipedia.org/wiki/Clustering_coefficient#Network_average_clustering_coefficient>`_
	for graph ``G``.
	"""
	cdef np.ndarray[np.float_t, ndim=1] C = lcc_(G)
	
	return <float> np.ma.masked_equal(C,-1.0).mean()