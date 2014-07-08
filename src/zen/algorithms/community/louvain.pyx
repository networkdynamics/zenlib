from zen.graph cimport Graph
from zen.digraph cimport DiGraph

cimport communityset as cs
import community_common as common

import numpy as np
cimport numpy as np
from cpython cimport bool

import sys

cdef sum_in_out(G, int n, np.ndarray counts, np.ndarray comms, bool weighted):
	cdef:
		float sum_in = 0.0
		float sum_out = 0.0
		float amt = 0.0

	for m in G.neighbors_(n):
		if weighted:
			amt = G.weight_(G.edge_idx_(n, m))
		else:
			amt = 1.0

		if comms[n] == comms[m]:
			sum_in += amt
		else:
			sum_out += amt

	return (sum_in, sum_out)

cdef initialize_counts(G, np.ndarray counts, np.ndarray comms, bool weighted):
	cdef int n
	for n in G.nodes_():
		in_out = sum_in_out(G, n, counts, comms, weighted)
		counts[n,0] += (in_out[0] / 2.0)
		counts[n,1] += (counts[n,0] + in_out[1])

cdef comm_add_node(G, int comm, int n, np.ndarray counts, np.ndarray comms,
					bool weighted):
	comms[n] = comm

	in_out = sum_in_out(G, n, counts, comms, weighted)
	
	counts[comm,0] += (in_out[0] / 2.0)
	counts[comm,1] += (counts[comm,0] + in_out[1])

cdef comm_remove_node(G, int n, np.ndarray counts, np.ndarray comms, bool weighted):

	cdef int comm = comms[n]	
	in_out = sum_in_out(G, n, counts, comms, weighted)
	
	counts[comm,0] -= (in_out[0] / 2.0)
	counts[comm,1] -= (counts[comm,0] + in_out[1])

	comms[n] = -1

cdef mod_gain(G, node, new_comm, num_edges_graph, weighted, 
			k, np.ndarray counts, np.ndarray comms):
	# Compute the modularity gain obtained by adding ``node`` to ``new_comm``.
	# For this, we used some cached values: the number of edges in the graph,
	# and the number / sum of weights incident to ``node`` (``k``).
	# The formula is straight from the paper

	cdef:
		float denom, a, b, c, d, e

	denom = 2.0 * num_edges_graph
	a = (counts[new_comm,0] + sum_incident(G, node, weighted, comms, new_comm)) / denom
	b = (counts[new_comm,1] + k) / denom
	c = counts[new_comm,0] / denom
	d = counts[new_comm,1] / denom
	e = k / denom

	return ((a - (b * b)) - (c - (d * d) - (e * e)))

cdef sum_incident(G, int node, bool weighted, np.ndarray comms, 
						int in_community=-1):
	# Sum of the edges (or edge weights) incident to a node. If in_community is
	# not -1, only account for neighbors which are part of that community.

	cdef:
		float total = 0.0
		float amt

		int m

	for m in G.neighbors_(node):
		if in_community != -1 and comms[m] != in_community:
			continue		

		amt = 1.0
		if weighted:
			amt = G.weight_(G.edge_idx_(node, m))
		total += amt

	return total

cdef optimize_modularity(G, float sum_edges, np.ndarray counts, 
								np.ndarray comms, bool weighted):
	# Optimize the modularity of the communities over the graph by moving nodes
	# to the neighbor's community which provides the greatest increase. This
	# goes on until no more moves are possible.

	cdef:
		bool moved = True # Has a node moved over this iteration
		bool improvement = False # Did we improve modularity over this iteration 

		float max_delta_mod
		float delta_mod

		float k # Sum of incident edges to a node

		# Node iterators
		int n
		int m

		int best_community
		int old_community
		int comm_m

	while moved:
		moved = False
		for n in G.nodes_():
			best_community = comms[n] # Best by default: no change
			old_community = comms[n]
			max_delta_mod = 0.0 # Minimal delta accepted: no change

			k = sum_incident(G, n, weighted, comms)

			comm_remove_node(G, n, counts, comms, weighted)
			for m in G.neighbors_(n):
				if m == n: #Ignore self-loops
					continue

				comm_m = comms[m]
				delta_mod = mod_gain(G, n, comm_m, sum_edges, weighted, k, 
									counts, comms)

				if delta_mod > max_delta_mod:
					max_delta_mod = delta_mod
					best_community = comms[m]
					

			if best_community != old_community:
				moved = True
				improvement = True

			comm_add_node(G, best_community, n, counts, comms, weighted)

	return improvement

cdef create_metagraph(old_graph, np.ndarray comms, bool weighted):
	# In the metagraph, nodes are communities from the old graph. Edges connect
	# communities whose nodes are connected in the old graph ; internal edges
	# in the old graph correspond to self-loops in the metagraph. Edges are
	# weighted by the sum of the edge weights they represent in the old graph.

	community_dict = {}

	cdef: 
		int n
		int neigh
		int cidx
		int neigh_cidx
		float amt

	G = Graph()

	for n in old_graph.nodes_():
		cidx = comms[n]
		if cidx not in community_dict:
			community_dict[cidx] = [n]
		else:
			community_dict[cidx].append(n)

	for cidx, comm in community_dict.iteritems():

		if cidx not in G:
			G.add_node(cidx)
		for n in comm:
			for neigh in old_graph.neighbors_(n):
				amt = 1.0
				if weighted:
					amt = old_graph.weight_(old_graph.edge_idx_(n, neigh))
				neigh_cidx = comms[neigh]

				if neigh_cidx not in G:
					G.add_node(neigh_cidx)

				if not G.has_edge(cidx, neigh_cidx):
					G.add_edge(cidx, neigh_cidx, None, amt)
				else:
					G.set_weight(cidx, neigh_cidx, G.weight(cidx, neigh_cidx) + amt)

	# Only self-loops count double
	for edge in G.edges_(-1, False, True):
		if G.endpoints_(edge[0])[0] != G.endpoints_(edge[0])[1]:
			G.set_weight_(edge[0], edge[1] / 2.0)

	return G

def louvain(G, **kwargs):
	"""
	Detect communities in a network using the Louvain algorithm described in
	[BLO2008]_. It assigns every node to its own community, and then tries to
	improve the modularity of the network by moving each node to the communities
	of its neighbors. Once no more increase is possible, a meta-network is built
	from these communities (the nodes being the communities themselves and the
	edges being the sum of the edges between members of these communities) and
	the process is repeated. This continues until no improvement in modularity
	is possible.

	**Keyword Args**

		* ``use_weights [=False]`` (bool): whether to take the weights of the
		network into account.

		* ``num_iterations [=None]`` (int): if not ``None``, the algorithm will
		stop after this many iterations of building meta-networks. This can be
		used to examine a community structure at different levels of resolution
		(i.e. a low number will return fine-grained communities, while a large
		number will return more general communities).

	**Returns**

		A :py:module:?CommunitySet containing the communities detected in the 
		graph.

	..[BLO2008]
		Blondel, V. et al 2008. Fast unfolding of communities in large networks.
			Journal of Statistical Mechanics, Vol. 2008, Issue 10.

	"""

	weighted = kwargs.pop("use_weights", False)
	num_iterations = kwargs.pop("num_iterations", None)

	sum_edges = 0.0
	if weighted:
		for e in G.edges_(weight=True):
			sum_edges += e[1]
	else:
		sum_edges = len(G.edges())

	comms = np.empty(G.max_node_idx + 1, dtype=np.int_)
	for i in range(G.max_node_idx + 1):
		comms[i] = i

	counts = np.zeros((G.max_node_idx + 1, 2), dtype=np.float_)
	initialize_counts(G, counts, comms, weighted)

	improved = optimize_modularity(G, sum_edges, counts, comms, weighted)
	
	# Initial "meta" values
	meta_comms = comms
	meta = G
	count_iter = 1
	while improved and (num_iterations is None or num_iterations < count_iter):	
		meta = create_metagraph(meta, meta_comms, weighted)
		weighted = True # The weight in metagraphs is always significant

		meta_comms = np.empty(meta.max_node_idx + 1, dtype=np.int_)
		for i in range(meta.max_node_idx + 1):
			meta_comms[i] = i

		meta_counts = np.zeros((meta.max_node_idx + 1, 2), dtype=np.float_)
		initialize_counts(meta, meta_counts, meta_comms, weighted)

		sum_edges = 0.0
		for e in meta.edges_(weight=True):
			sum_edges += e[1]

		improved = optimize_modularity(meta, sum_edges, meta_counts,
										meta_comms, weighted)

		for n in G.nodes_():
			comm = comms[n]
			meta_idx = meta.node_idx(comm)
			comms[n] = meta_comms[meta_idx]

		count_iter += 1

	num_communities = common.normalize_communities(comms)
	return cs.CommunitySet(G, comms, num_communities)

