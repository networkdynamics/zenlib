"""
This module implements some of the simple network generation functions.
"""
from zen.graph cimport Graph
from zen.digraph cimport DiGraph
from libc.stdlib cimport RAND_MAX, rand, srand
from cpython cimport bool

__all__ = ['erdos_renyi','barabasi_albert']

def barabasi_albert(n, m, **kwargs):
	"""
	Generate a random graph using the Barabasi-Albert preferential attachment model.
	
	Required parameters:
	  - n - the number of nodes to add to the graph
	  - m - the number of edges a new node will add to the graph
	
	Optional parameters:
	  -  seed - an integer seed for the random number generator
	
	Citation:
	A. L. Barabási and R. Albert "Emergence of scaling in random networks", Science 286, pp 509-512, 1999.
	"""
	seed = kwargs.pop('seed',0)
	
	return __inner_barabasi_albert(n, m, seed)
	
def identity_fxn(i):
	return i	
	
cdef __inner_barabasi_albert(int n, int m, int seed):
	
	cdef Graph G = Graph()
	cdef int new_node_idx, i, e
	cdef int rnd
	cdef int num_endpoints
	cdef int running_sum
	cdef bool edge_made
	
	# add nodes
	G.add_nodes(n, identity_fxn)
	
	#####
	# add edges
	srand(seed)
	
	# add the first (m+1)th node
	for i in range(m):
		G.add_edge_(i,m)
	
	# add the remaining nodes
	num_endpoints = 2 * m
	for new_node_idx in range(m+1,n):
		
		# this node drops m edges
		delta_endpoints = 0
		for e in range(m):
			rnd = rand() % (num_endpoints-delta_endpoints)
			
			# now loop through nodes and find the one whose endpoint has the running sum
			# note that we ignore nodes that we already have a connection to
			running_sum = 0
			for i in range(new_node_idx):
				if G.has_edge_(new_node_idx,i):
					continue
					
				running_sum += G.node_info[i].degree
				if running_sum > rnd:
					G.add_edge_(new_node_idx,i)
					delta_endpoints += G.node_info[i].degree - 1
					break
					
		num_endpoints += m * 2
	return G

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