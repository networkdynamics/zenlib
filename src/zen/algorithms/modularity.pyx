"""
The ``zen.algorithms.modularity`` module implements functions concerning the quantification of the presence of sub-groupings
of nodes in the network.  Functions in the module are available by importing the root-level ``zen`` package.

.. autofunction:: modularity(G,C,weighted=False)
"""

import numpy as np
cimport numpy as np
from zen.graph cimport Graph
from zen.exceptions import ZenException

__all__ = ['modularity']

# calls modularity_
# communities is a dictionary of communities
# It assumes that the communities are non-overlapping, 
# So, if overlapping communities are there, the output will be not-meaningful

cpdef double total_weight(Graph graph):
	cdef double tweight=0.0
	cdef int next_node
	cdef int degree,eidx,ei
	for next_node in graph.nodes_():
		#
		degree = graph.node_info[next_node].degree
		#d_c+=degree
		for ei in range(degree):
			eidx = graph.node_info[next_node].elist[ei]
			tweight+=graph.weight_(eidx)
	tweight/=2
	return(tweight)
		
	
def modularity(G, C, **kwargs):
	"""
	Compute the modularity of the groupings of nodes in ``C``.
	
	.. warning::
		This function will be revised very soon.  Expect changes.
	
	**Args**:
		* ``G`` (:py:class:`zen.Graph`): the undirected graph on which to compute the modularity of the groupings of nodes provided.
		* `C` (:py:class:`dict`): the group assignment of nodes in ``G``.  C[i] is the set of nodes which belong to group ``i``.
		
	**KwArgs**:
		* ``weighted [=False]`` (boolean): whether or not the weight of edges should influence the degree to which directly
		  connected nodes are associated with one another.
	"""
	weighted = kwargs.pop('weighted',False)
	
	if type(G) != Graph:
		raise ZenException, 'Only graphs of type zen.Graph are supported'
	
	return _modularity(G,C,weighted)
	
cpdef double _modularity(graph, communities, weighted):
	cdef double modularity_value=0.0
	cdef int i
	cdef int upper_limit = graph.max_node_idx + 1
	cdef np.ndarray[np.int_t, ndim=1] node_indices
	if type(communities) is not dict:
		raise ZenException, 'communities should be dictionary keyed by community id and valued by node objects'
	
	node_collections=communities.values()
	cdef np.ndarray[np.int_t, ndim=1] node_assignment = np.ones(upper_limit, np.int)*-1
	cdef int community_idx=0
	cdef double tweight
	if weighted==True:
		tweight=total_weight(<Graph> graph)
		
	for nodes in node_collections: # nodes is a community
		node_indices = np.zeros(len(nodes), np.int)
    
    	# Loop through the list of nodes, get each index
		i = 0
		for node in nodes:
			node_indices[i] = graph.node_idx(node)
			node_assignment[node_indices[i]]=community_idx
			i = i + 1
		if(weighted==False):
			modularity_value+=subgraph_modularity(<Graph> graph, node_indices,node_assignment)
		else:
			
			modularity_value+=subgraph_modularity_weighted(<Graph> graph, node_indices,node_assignment,<double>tweight)
			
		community_idx+=1
		
	return(modularity_value)
#
cpdef double subgraph_modularity_weighted(Graph graph, np.ndarray[np.int_t, ndim=1] node_indices, np.ndarray[np.int_t, ndim=1] node_assignment, double tweight):
	cdef double l_c=0.0
	cdef double d_c=0.0
	cdef double m = tweight
	cdef double Q_subgraph
	cdef int next_node
	cdef int degree,ei,eidx,j
	# Faiyaz print 'In subgraph modularity'
	
	for next_node in node_indices:
		degree = graph.node_info[next_node].degree
		#d_c+=degree
		for ei in range(degree):
			eidx = graph.node_info[next_node].elist[ei]
			d_c+=graph.weight_(eidx)
			j = graph.endpoint_(eidx,next_node)
			if(node_assignment[j]==node_assignment[next_node]):
				#l_c+=1
				l_c+=graph.weight_(eidx)
	l_c/=2
	# Faiyaz print 'l_c:'+ str(l_c)
	# Faiyaz print 'd_c:'+ str(d_c)
	
	Q_subgraph=l_c*1.0/m-(d_c*1.0/(2*m))*(d_c*1.0/(2*m))
	return(Q_subgraph)
	



	
cpdef double subgraph_modularity(Graph graph, np.ndarray[np.int_t, ndim=1] node_indices, np.ndarray[np.int_t, ndim=1] node_assignment):
	cdef int l_c=0
	cdef int d_c=0
	cdef int m = graph.num_edges
	cdef double Q_subgraph
	cdef int next_node
	cdef int degree,ei,eidx,j
	# Faiyaz print 'In subgraph modularity'
	
	for next_node in node_indices:
		degree = graph.node_info[next_node].degree
		d_c+=degree
		for ei in range(degree):
			eidx = graph.node_info[next_node].elist[ei]
			j = graph.endpoint_(eidx,next_node)
			if(node_assignment[j]==node_assignment[next_node]):
				l_c+=1
	l_c/=2
	# Faiyaz print 'l_c:'+ str(l_c)
	# Faiyaz print 'd_c:'+ str(d_c)
	
	Q_subgraph=l_c*1.0/m-(d_c*1.0/(2*m))*(d_c*1.0/(2*m))
	return(Q_subgraph)
	
'''				

# nodes = iterable, containing indices to nodes
cpdef double modularity_(graph, nodes):
    if type(graph) is not Graph:
        raise ZenException, 'Only Graph objects are supported in modularity'

    cdef int num_nodes = len(nodes)
    cdef np.ndarray[np.int_t, ndim=1] nodes_array
    cdef int i = 0

    # Check if the list of nodes is already a numpy array
    # If it is not, convert it to one
    if type(nodes) is not np.ndarray:
        # Create a numpy array, initialise it with zeros
        nodes_array = np.zeros(num_nodes, np.int)
    
        # Now fill the array with values
        for node in nodes:
            nodes_array[i] = node
            i = i + 1
    else:
        # It's already a numpy array
        nodes_array = nodes

    cdef float sum = 0
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            sum = sum + Q_(graph, nodes_array[i], nodes_array[j])
    return sum

# Change to cdef later - for testing    
cdef float Q_(Graph graph, int node_1, int node_2):
    cdef int degree_1 = graph.degree_(node_1)
    cdef int degree_2 = graph.degree_(node_2)
    cdef int total_edges = graph.num_edges
    cdef float expected = float(degree_1 * degree_2) / (2 * total_edges)

    cdef float actual = 1 if graph.has_edge_(node_1, node_2) else 0
    cdef float Q_total = float(actual - expected) / float(2 * total_edges)
    return Q_total
'''