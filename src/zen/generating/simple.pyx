"""
This module implements some of the simple network generation functions.
"""
from zen.graph cimport Graph
from zen.digraph cimport DiGraph
from libc.stdlib cimport RAND_MAX, rand

__all__ = ['erdos_renyi']

cpdef erdos_renyi(int num_nodes,float p,bint directed=False,bint self_loops=False):
	"""
	Generate an erdos-renyi graph with num_nodes nodes, each edge existing with probability p.
	"""
	if directed:
		return __erdos_renyi_directed(num_nodes,p,self_loops)
	else:
		return __erdos_renyi_undirected(num_nodes,p,self_loops)

cpdef __erdos_renyi_undirected(int num_nodes,float p,bint self_loops):
	cdef Graph G = Graph()
	cdef int i, j, first_j
	cdef float rnd
	
	# add nodes
	for i in range(num_nodes):
		G.add_node(i)
		
	# add edges
	for i in range(num_nodes):
		if self_loops:
			first_j = i
		else:
			first_j = i+1
			
		for j in range(first_j,num_nodes):
			rnd = rand()
			rnd = rnd / (<float> RAND_MAX)
			if rnd < p:
				G.add_edge_(i,j)
	
	return G
	
cpdef __erdos_renyi_directed(int num_nodes,float p,bint self_loops):
	cdef DiGraph G = DiGraph()
	cdef int i, j
	cdef float rnd

	# add nodes
	for i in range(num_nodes):
		G.add_node(i)
	
	# add edges
	for i in range(num_nodes):
		for j in range(num_nodes):
			if i == j and not self_loops:
				continue
				
			rnd = rand()
			rnd = rnd / (<float> RAND_MAX)
			if rnd < p:
				G.add_edge_(i,j)

	return G	
				
			
	