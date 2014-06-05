__all__ = [ 'label_propagation' ]

from zen.graph cimport Graph
from zen.digraph cimport DiGraph
from zen import ZenException

cimport communityset as cs
import community_common as common

import numpy as np
cimport numpy as np

from cpython cimport bool


# Return a map containing the number of occurences of labels
# among the neighbors of a node
cdef count_neighbor_lbls(G, int node, np.ndarray[np.int_t] labels,
						bool use_weights):

	# TODO: Can we ditch the dictionary in favor of np arrays here?

	neighbor_lbl_counts = {}
	for nghbr in G.neighbors_(node):
		# Update the count for this neighbor
		nghbr_lbl = labels[nghbr]

		if use_weights:
			amt = G.weight_(G.edge_idx_(node, nghbr))
		else:
			amt = 1

		if nghbr_lbl not in neighbor_lbl_counts:
			neighbor_lbl_counts[nghbr_lbl] = amt
		else:
			neighbor_lbl_counts[nghbr_lbl] += amt			

	return neighbor_lbl_counts

# Checks if the LPA should be stopped given the found label. This is the case 
# if every node's label is shared by the majority of its neighbors.
cdef should_stop_lpa(G, np.ndarray[np.int_t] labels, bool use_weights):
	cdef np.ndarray[np.int_t] nodes = G.nodes_()
	for node in nodes:
		neighbor_lbl_counts = count_neighbor_lbls(G, node, labels, use_weights)
		keys = common.keys_of_max_value(neighbor_lbl_counts)

		if labels[node] not in keys:
			return False

	return True

def label_propagation(G, **kwargs):
	"""
	Detect communities in a graph using the Label-Propagation Algorithm (LPA)
	described in [RAG2007]_. It assigns a unique label to each node, then 
	propagates the labels by assigning to each node the label shared by the
	majority of its neighbors (ties are broken randomly). This continues until
	each node's label agrees with its neighbors'. Communities are the groups of
	nodes with the same label.

	**Keyword Args**:

		* ``use_weights [=False]`` (boolean): if ``True``, then the weights of
			the graph are taken into consideration when detecting communities.
		
		* ``max_iterations [=None]`` (int): if greater than or equal to zero, 
			the algorithm will run at most this many iterations. Negative values
			and ``None`` indicate that the algorithm will run until normal 
			completion.
			
	**Returns**:
		A :py:class:`CommunitySet` containing the communities detected in the 
		graph.

	..[RAG2007] 
		Raghavan, U. N., Albert, R. and Kumara, S. 2007. Near linear time 
			algorithm to detect community structures in large-scale networks. 
			Physical Review E, Vol. 76, No. 3.

	"""
	
	if type(G) != Graph and type(G) != DiGraph:
		raise ZenException, 'Unknown graph type: %s' % type(G)

	use_weights = kwargs.pop('use_weights', False)
	max_iterations = kwargs.pop('max_iterations', None)
	if max_iterations is None:
		max_iterations = -1

	cdef int i = 0

	cdef np.ndarray[np.int_t] nodes = G.nodes_()

	# Initialize each node as having its own label
	rnge = G.max_node_idx + 1
	cdef np.ndarray[np.int_t] label_table = np.arange(rnge)

	while True:
		if max_iterations >= 0 and i == max_iterations:
			break

		# At each iteration, look at nodes in a random order
		np.random.shuffle(nodes)

		# Propagate labels
		for node in nodes:
			lbl_counts = count_neighbor_lbls(G, node, label_table, use_weights)
			
			# Select new label randomly among those that have maximal count
			keys = common.keys_of_max_value(lbl_counts)
			if not keys is None:
				label_table[node] = keys[np.random.randint(len(keys))]

		if should_stop_lpa(G, label_table, use_weights):
			break

		i += 1

	num_communities = common.normalize_communities(label_table)
	return cs.CommunitySet(G, label_table, num_communities)
