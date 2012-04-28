"""
The ``zen.algorithms.matching`` module provides routines for computing `maximum-matchings <???>`_ on various types of graphs.

Functions
---------

.. autofunction:: maximum_matching

.. autofunction:: maximum_matching_

.. autofunction:: hopcroft_karp_
"""

from zen.bipartite cimport BipartiteGraph
from zen.digraph cimport DiGraph
from zen.exceptions import *
import numpy as np
cimport numpy as np
from Queue import Queue

__all__ = ['maximum_matching','maximum_matching_','hopcroft_karp_']

# TODO(druths): Add support for matching undirected networks

def maximum_matching(G):
	"""
	Find a set of edges that comprise a maximum-matching for the graph ``G``.
	
		* If the graph is a bipartite graph (:py:class:`zen.BipartiteGraph`), then the standard bipartite maximum-matching
		  problem is solved using the Hopcroft-Karp algorithm.
		* If the graph is a directed graph (:py:class:`zen.DiGraph`), then the edge subset is found such that no two edges
		  in the subset share a common starting vertex or a common ending vertex.
		* If the graph is undirected, an error is thrown as this is not currently supported.
	
	**Returns**:
		:py:class:`list`. The list of edge endpoint pairs that comprise the edges belonging to a maximum-matching for the graph.
		
	**Raises**:
		:py:exc:`zen.ZenException`: if ``G`` is an undirected graph.
	"""
	eidx_matching = maximum_matching_(G)
	
	return [G.endpoints(eidx) for eidx in eidx_matching]

cpdef maximum_matching_(G):
	"""
	Find a set of edges that comprise a maximum-matching for the graph ``G``.

		* If the graph is a bipartite graph (:py:class:`zen.BipartiteGraph`), then the standard bipartite maximum-matching
		  problem is solved using the Hopcroft-Karp algorithm.
		* If the graph is a directed graph (:py:class:`zen.DiGraph`), then the edge subset is found such that no two edges
		  in the subset share a common starting vertex or a common ending vertex.
		* If the graph is undirected, an error is thrown as this is not currently supported.

	**Returns**:
		:py:class:`list`. The list of edge indices that indicate the edges belonging to a maximum-matching for the graph.
	
	**Raises**:
		:py:exc:`zen.ZenException`: if ``G`` is an undirected graph.
	"""
	if type(G) == BipartiteGraph:
		return __bipartite_hopcroft_karp_(<BipartiteGraph>G)
	elif type(G) == DiGraph:
		return __directed_hopcroft_karp_(<DiGraph>G)
	else:
		raise ZenException, 'Only bipartite and directed graphs are currently supported'

def hopcroft_karp_(G):
	"""
	Find a set of edges that comprise a maximum-matching for the bipartite graph ``G`` using the `Hopcroft-Karp algorithm <???>`_.

	**Returns**:
		:py:class:`list`. The list of edge indices that indicate the edges belonging to a maximum-matching for the graph.
	
	**Raises**:
		:py:exc:`zen.ZenException`: if ``G`` is not a bipartite graph.
	"""
	if type(G) == BipartiteGraph:
		return __bipartite_hopcroft_karp_(<BipartiteGraph>G)
	else:
		raise ZenException, 'Only bipartite graphs are currently supported'

cpdef __directed_hopcroft_karp_(DiGraph G):
	
	cdef int unode, vnode, i
	
	#####
	# apply the transformation to produce a bipartite graph
	GT = BipartiteGraph()
	tnode2node = {}
	node2unode = {}
	node2vnode = {}
	
	# add the nodes
	for i in G.nodes_iter_():
		unode = GT.add_u_node()
		vnode = GT.add_v_node()
		tnode2node[unode] = i
		tnode2node[vnode] = i
		node2unode[i] = unode
		node2vnode[i] = vnode
	
	# add the edges
	for i in G.edges_iter_():
		u,v = G.endpoints_(i)
		#print u,node2unode[u],GT.is_in_U_(node2unode[u])
		#print v,node2vnode[v],GT.is_in_U_(node2vnode[v])
		GT.add_edge_(node2unode[u],node2vnode[v],i)
	
	#####
	# run the bipartite matching
	max_matching = __bipartite_hopcroft_karp_(GT)
	
	#####
	# transform the maximum matching back into the directed graph
	di_max_matching = [GT.edge_data_(i) for i in max_matching]
	
	return di_max_matching
		
cpdef __bipartite_hopcroft_karp_(BipartiteGraph G):

	cdef int NIL_V = G.next_node_idx
	cdef np.ndarray[np.int_t, ndim=1] pairs = np.ones(G.next_node_idx+1, np.int) * NIL_V
	cdef np.ndarray[np.int_t, ndim=1] layers = np.ones(G.next_node_idx+1, np.int) * -1
	cdef np.ndarray[np.int_t, ndim=1] u_nodes = G.U_()
	matching_list = []
	
	while __bhk_bfs(G,pairs,layers,u_nodes):
		for v in u_nodes:
			if pairs[v] == NIL_V:
				__bhk_dfs(G,v,pairs,layers)
	
	# construct the matching list
	for u in u_nodes:
		if pairs[u] != NIL_V:
			matching_list.append(G.edge_idx_(u,pairs[u]))
	
	return matching_list

cdef __bhk_bfs(BipartiteGraph G, np.ndarray[np.int_t, ndim=1] pairs, np.ndarray[np.int_t, ndim=1] layers, np.ndarray[np.int_t,ndim=1] u_nodes):
	cdef int NIL_V = G.next_node_idx
	cdef int i, v
	Q = Queue()
	
	for v in u_nodes:
		if pairs[v] == NIL_V:
			layers[v] = 0
			Q.put(v)
		else:
			layers[v] = -1
	layers[NIL_V] = -1
	
	while not Q.empty():
		v = Q.get()

		if v is NIL_V:
			continue
			
		for i in range(G.node_info[v].degree):
			u = G.endpoint_(G.node_info[v].elist[i],v)
			if layers[pairs[u]] == -1:
				layers[pairs[u]] = layers[v] + 1
				Q.put(pairs[u])
				
	return layers[NIL_V] != -1
	
cdef __bhk_dfs(BipartiteGraph G, int v, np.ndarray[np.int_t, ndim=1] pairs, np.ndarray[np.int_t, ndim=1] layers):
	cdef int NIL_V = G.next_node_idx
	cdef int i
	
	if v != NIL_V:
		for i in range(G.node_info[v].degree):
			u = G.endpoint_(G.node_info[v].elist[i],v)
			if layers[pairs[u]] == layers[v] + 1:
				if __bhk_dfs(G, pairs[u], pairs, layers):
					pairs[u] = v
					pairs[v] = u
					return True
		layers[v] = -1
		return False
	return True
	