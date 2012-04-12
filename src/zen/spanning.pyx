#cython: embedsignature=True

"""
This module provides functions that calculate spanning trees of a network.
"""

from graph cimport Graph
from digraph cimport DiGraph
from zen.util.fiboheap cimport FiboHeap
import numpy as np
cimport numpy as np

from cpython cimport bool

__all__ = ['minimum_spanning_tree']

cpdef minimum_spanning_tree(Graph G):
	"""
		This will return the minimum spanning tree (MST) of Graph G, based on Prim's Algorithm.
		The algorithm was taken from;
		Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. Introduction to Algorithms, Second Edition. 
		MIT Press and McGraw-Hill, 2001. ISBN 0-262-03293-7. Chapter 23
	"""
	cdef Graph Gnew = Graph()
	cdef np.ndarray[object, ndim=1] fiboheap_nodes = np.empty([G.num_nodes], dtype=object) # holds all of our FiboHeap Nodes Pointers
	cdef np.ndarray[np.int_t, ndim=1] node_parents = np.empty([G.num_nodes], dtype=np.int) # holds the parent index for each node.
	cdef np.ndarray[np.int_t, ndim=1] node_mapping = np.empty([G.num_nodes], dtype=np.int) # Mapping old nodes to new nodes
	cdef np.ndarray[np.uint8_t, ndim=1] Vorig = np.empty([G.num_nodes], dtype=np.uint8) # "Boolean" array for nodes that are originally in set V (Graph G)
	cdef double infinity = float('infinity')
	cdef bool firstnode = True
	cdef int parent, new_parent, i, u, v, num_edges
	cdef float v_key, vdis
	
	Q = FiboHeap()

	# For readability. Apparently, no such thing as a np.bool type. Using unsigned int8.
	cdef np.uint8_t TRUE = np.uint8(1)
	cdef np.uint8_t FALSE = np.uint8(0)
	
	# Go through all the nodes in the original Graph and add them to the FiboHeap with a key of infinity.
	# The first node is inserted right away in the new Graph, and in the FiboHeap with a key of zero.
	for i in range(G.next_node_idx):
		if G.node_info[i].exists:
			if firstnode:
				node_mapping[i] = Gnew.add_node(G.node_object(i), G.node_data_(i))
				Vorig[i] = FALSE
				fiboheap_nodes[i]=Q.insert(0.0, i)
				firstnode = False
			else:
				Vorig[i] = TRUE
				fiboheap_nodes[i]=Q.insert(infinity, i)
	
	while Q.get_node_count() != 0:
		u = Q.extract()
		if Vorig[u] == TRUE: # This will be TRUE for everything but the root, which was added earlier.
			# Add the node to the graph, specifying the parent that was stored earlier.
			parent = node_parents[u]
			new_parent = node_mapping[parent]
			node_mapping[u] = Gnew.add_node(G.node_object(u), G.node_data_(u))
			Gnew.add_edge_(node_mapping[u], new_parent, G.edge_data_(G.edge_idx_(u, parent)), G.weight_(G.edge_idx_(u,parent)))
			Vorig[u] = FALSE
		# If neighbor v is still not in the new Graph, and its "key" can be decreased based on new node u, do it.
		# loop over in edges
		num_edges = G.node_info[u].degree
		elist = G.node_info[u].elist
		for i in range(num_edges):
			v = G.edge_info[elist[i]].u
			if v == u:
				v = G.edge_info[elist[i]].v
		
			if Vorig[v] == TRUE:
				v_key = Q.get_node_key(<object>fiboheap_nodes[v])
				v_dis = G.weight_(G.edge_idx_(u,v))
				if v_dis < v_key:
					node_parents[v] = u
					Q.decrease_key(<object>fiboheap_nodes[v],v_dis)
	return Gnew
