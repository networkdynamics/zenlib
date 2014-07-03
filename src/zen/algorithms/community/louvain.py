from zen.graph import Graph

import communityset as cs
import community_common as common

import numpy as np

class LouvainCommunity:
	# Internal community object -- mainly used to cache incident and internal
	# edge counts (so as to not re-calculate those counts on every node move)

	def __init__(self, G, node_ids, weighted):
		self._nodes = node_ids
		self._weighted = weighted
		self._G = G
		
		self._sum_in = 0 # Sum of links inside the community
		self._sum_total = 0 # Sum of links incident to the community
		sum_out = 0
		for n in self._nodes:
			in_out = self.__sum_in_out(n)
			self._sum_in += in_out[0]
			sum_out += in_out[1]

		self._sum_in /= 2 # If we added to in, we counted twice
		self._sum_total = self._sum_in + sum_out
	
	def __contains__(self, n):
		return n in self._nodes

	def __iter__(self):
		return iter(self._nodes)

	def __len__(self):
		return len(self._nodes)

	def __sum_in_out(self, n):
		sum_in = 0
		sum_out = 0
		for m in self._G.neighbors_(n):
				amt = self._G.weight_(self._G.edge_idx_(n,m))
				if m in self._nodes:
					sum_in += amt
				else:
					sum_out += amt
		return (sum_in, sum_out)

	def sum_in(self):
		return self._sum_in

	def sum_total(self):
		return self._sum_total

	def add_node(self, n):
		assert n not in self._nodes

		self._nodes.add(n)

		delta_in, delta_out = self.__sum_in_out(n)

		self._sum_in += delta_in
		self._sum_total += (delta_in + delta_out)

	def remove_node(self, n):
		assert n in self._nodes

		self._nodes.remove(n)
		
		delta_in, delta_out = self.__sum_in_out(n)

		self._sum_in -= delta_in
		self._sum_total -= (delta_in + delta_out)

def mod_gain(G, node, new_comm, num_edges_graph, weighted, k):
	# Compute the modularity gain obtained by adding ``node`` to ``new_comm``.
	# For this, we used some cached values: the number of edges in the graph,
	# and the number / sum of weights incident to ``node`` (``k``).
	# The formula is straight from the paper

	denom = 2.0 * num_edges_graph
	a = (new_comm.sum_in() + sum_incident(G, node, weighted, new_comm)) / denom
	b = (new_comm.sum_total() + k) / denom
	c = new_comm.sum_in() / denom
	d = new_comm.sum_total() / denom
	e = k / denom

	return ((a - (b * b)) - (c - (d * d) - (e * e)))

def sum_incident(G, node, weighted, in_community=None):
	# Sum of the edges (or edge weights) incident to a node. If in_community is
	# not None, only account for neighbors which are part of that community.

	total = 0.0
	for m in G.neighbors_(node):
		if in_community != None and m not in in_community:
			continue		

		amt = 1.0
		if weighted:
			amt = G.weight_(G.edge_idx_(node, m))
		total += amt

	return total

def optimize_modularity(G, sum_edges, communities, assignments, weighted):
	# ``communities`` is a dictionary of LouvainCommunity objects keyed by 
	# community index. ``assignments`` is a list of nodes indices to community 
	# indices.

	# Optimize the modularity of the communities over the graph by moving nodes
	# to the neighbor's community which provides the greatest increase. This
	# goes on until no more moves are possible.

	moved = True
	improvement = False

	while moved:
		moved = False
		for n in G.nodes_():
			best_community = assignments[n] # Best by default: no change
			max_delta_mod = 0.0

			k = sum_incident(G, n, weighted)

			communities[assignments[n]].remove_node(n)
			for m in G.neighbors_(n):
				comm_m = communities[assignments[m]]
				delta_mod = mod_gain(G, n, comm_m, sum_edges, weighted, k)

				if delta_mod > max_delta_mod:
					max_delta_mod = delta_mod
					best_community = assignments[m]

			if best_community != assignments[n]:
				moved = True
				improvement = True

			assignments[n] = best_community
			communities[assignments[n]].add_node(n)

	return improvement

def create_metagraph(old_graph, communities, assignments, weighted):
	# In the metagraph, nodes are communities from the old graph. Edges connect
	# communities whose nodes are connected in the old graph ; internal edges
	# in the old graph correspond to self-loops in the metagraph. Edges are
	# weighted by the sum of the edge weights they represent in the old graph.

	G = Graph()

	for cidx, comm in communities.iteritems():
		if len(comm) == 0:
			continue

		if cidx not in G:
			G.add_node(cidx)
		for node in comm:
			for neigh in old_graph.neighbors_(node):
				amt = 1.0
				if weighted:
					amt = old_graph.weight_(old_graph.edge_idx_(node, neigh))
				neigh_cidx = assignments[neigh]

				if neigh_cidx not in G:
					G.add_node(neigh_cidx)

				if not G.has_edge(cidx, neigh_cidx):
					G.add_edge(cidx, neigh_cidx, weight=amt)
				else:
					G.set_weight(cidx, neigh_cidx, G.weight(cidx, neigh_cidx) + amt)

	# Only self-loops count double
	for edge in G.edges_(weight=True):
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

	assignments = [i for i in range(G.max_node_idx + 1)]
	communities = {i: LouvainCommunity(G, set([i]), weighted) 
						for i in range(G.max_node_idx + 1)}

	improved = optimize_modularity(G, sum_edges, communities, assignments, weighted)
	
	count_iter = 1
	# Initial "meta" values
	meta_assignments = assignments
	meta_communities = communities
	meta = G
	while improved and (num_iterations is None or num_iterations < count_iter):
		meta = create_metagraph(meta, meta_communities, meta_assignments, weighted)
		weighted = True # The weight in metagraphs is always significant

		meta_assignments = [i for i in range(meta.max_node_idx + 1)]
		meta_communities = {i: LouvainCommunity(meta, set([i]), weighted) 
								for i in range(meta.max_node_idx + 1)}

		sum_edges = 0.0
		for e in meta.edges_(weight=True):
			sum_edges += e[1]

		improved = optimize_modularity(meta, sum_edges, meta_communities,
										meta_assignments, weighted)

		for nidx, comm in enumerate(assignments):
			meta_idx = meta.node_idx(comm)
			assignments[nidx] = meta_assignments[meta_idx]

		count_iter += 1

	num_communities = common.normalize_communities(assignments)
	return cs.CommunitySet(G, np.asarray(assignments), num_communities)

