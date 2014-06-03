from zen.graph cimport Graph
from zen.graph import ZenException
import communityset as cs
import community_common as common

cimport numpy as np
import numpy as np

from collections import deque

# Obtain the max eigenvalue and its associated eigenvector
# from a real symmetric matrix
cdef get_max_eigen(np.ndarray mat):
	vals, vects = np.linalg.eigh(mat)

	if len(vals) == 0:
		return None

	cdef int max_i = np.argmax(vals)
	return (vals[max_i], vects[:, max_i])

# Create a community vector from an eigenvector. Based on the eigenvector
# signs, the community vector entries are either 1 (first community) or -1
# (second community)
cdef np.ndarray compute_community_vector(np.ndarray eigenvec):
	cdef np.ndarray comm_vec = np.empty(eigenvec.shape[0], np.int)
	cdef int i
	for i in range(eigenvec.shape[0]):
		if(eigenvec[i] > 0):
			comm_vec[i] = 1
		else:
			comm_vec[i] = -1

	return comm_vec

# Computes the modularity of a sub-graph divided in two communities, 
# using its modularity matrix.
cdef float modularity(np.ndarray mod_mtx, np.ndarray comm_vec, int num_edges):
	return (1.0 / (4 * num_edges)) * np.dot(comm_vec, np.dot(mod_mtx, comm_vec))

# Modifies the communities "by hand" to get a better modularity.
cdef fine_tune_modularity(np.ndarray mod_matrix, np.ndarray comm_vector, 
							float initial_mod, int num_edges):
	cdef:
		float prev_mod = initial_mod
		set to_move # Nodes to move to the opposite community in this iteration
		float current_mod # Modularity at each step, after moving a node
		float max_mod_delta # Modularity gain by moving the optimal node
		int max_idx # Index of the optimal node to move
		int best_mod_idx # Best modularity achieved while moving nodes

		# Each index in the arrays represents a time-step t, in which moves[t]
		# is the node that was moved and moves_mods[t] is the modularity
		# achieved by moving nodes from 0 to t to their opposite community.
		np.ndarray[np.int_t] moves = np.empty(len(comm_vector), dtype=np.int_)
		np.ndarray[np.float_t] moves_mods = np.empty(len(comm_vector), dtype=np.float_)
		int move_i = 0

	while True:
		next_comm_vector = comm_vector.copy()
		to_move = set(range(len(comm_vector)))
		current_mod = prev_mod
		move_i = 0

		while len(to_move) > 0:
			max_mod_delta = -2
			max_idx = -1
	
			# Test which of the nodes that we can move will give us the best
			# result for modularity
			for i in to_move:
				# Move to the other community, compute modularity			
				next_comm_vector[i] *= -1
				
				# Record if we have better modularity with this node 
				mod_delta = (modularity(mod_matrix, next_comm_vector, num_edges)
							- prev_mod)
	
				if(mod_delta >= max_mod_delta):
					max_idx = i
					max_mod_delta = mod_delta

				# Move back to test the next node
				next_comm_vector[i] *= -1
		
			# Move the node found in the previous step
			to_move.remove(max_idx)
			next_comm_vector[max_idx] *= 1

			# Record this in the movement list
			current_mod += max_mod_delta
			moves[move_i] = max_idx
			moves_mods[move_i] = current_mod
			move_i += 1
			#movements.append((max_idx, current_mod))

		# Now that all nodes have been moved, select the state which had the
		# best modularity.
		best_mod_idx = 0
		for i in range(1, len(moves_mods)):
			if(moves_mods[i] > moves_mods[best_mod_idx]):
				best_mod_idx = i

		# If we could not improve our overall modularity, we're done
		if moves_mods[best_mod_idx] < prev_mod:
			break
		
		# Otherwise replicate the movements on the community vector. Only
		# replicate those that led to the best increase in modularity
		prev_mod = moves_mods[best_mod_idx]
		for i in range(best_mod_idx + 1):
			comm_vector[moves[i]] *= -1

		# Repeat this process until we cannot improve modularity further.

	#Nothing to return, we modified the community vector in-place

# Modifies the adjacency matrix to make it a modularity matrix.
cdef as_modularity_matrix(np.ndarray adj_mat, Graph G):
	cdef:
		int i
		int j
		float factor

	for i in range(adj_mat.shape[0]):
		for j in range(adj_mat.shape[1]):
			factor = (G.degree_(i) * G.degree_(j)) / (2.0 * G.size())
			adj_mat[i,j] -= factor

# Compute a modularity matrix for a part of the graph
cdef group_modularity_matrix(np.ndarray mod, list group):
	cdef:
		np.ndarray mod_mtx_g = np.empty((len(group), len(group)))
		int i
		int j
	
	for i in range(len(group)):
		for j in range(len(group)):
			mod_mtx_g[i,j] = mod[group[i],group[j]]

	for i in range(len(group)):
		mod_mtx_g[i,i] -= np.sum(mod_mtx_g[:,i])

	return mod_mtx_g

def spectral_modularity(G, **kwargs):
	"""
	Detect communities in a graph using the algorithm described in [NEW2006]_.
	It repeatedly divides the graph in two, using the leading eigenvector of
	the graph's modularity matrix to make the division. A sub-graph is
	considered indivisible if dividing it would result in negative modularity.
	This algorithm only supports undirected, unweighted graphs. In addition,
	the graph must be compact.

	** Keyword Args **

		* ``fine_tune [=False]`` (boolean): Whether to fine-tune the results at
		each step. If ``True``, the algorithm will manually move nodes from
		one community to the other after dividing a sub-graph in two, in order
		to maximize the modularity of the sub-graph. Hence, the detected
		communities should be of higher quality, but this comes at the expense
		of additional processing time.

	** Raises **
		``ZenException``: If the graph is directed, weighted or not compact.
        
    ..[NEW2006] 
        Newman, M. E. J. 2006. Modularity and community structure in networks.
			Proc. National Academy of Sciences, Vol. 103, No. 23.
        
	"""
	if G.is_directed():
		raise ZenException("This algorithm only supports undirected graphs.")
	if not G.is_compact():
		raise ZenException("The graph must be compact.")

	cdef int i

	edges = G.edges_(weight=True)
	for i in range(len(edges)):
		if edges[i,1] != 1:
			raise ZenException("This algorithm only supports unweighted graphs.")

	fine_tune = kwargs.pop("fine_tune", False);

	# Empty graph: no communities
	if len(G) == 0:
		return cs.CommunitySet(G, [], 0)

	# TODO This whole algorithm is nearly repeated twice - once out of the loop
	# and once in. Could it be possible to refactor this?

	cdef np.ndarray mod_mtx = G.matrix()
	as_modularity_matrix(mod_mtx, G)

	max_eigen = get_max_eigen(mod_mtx)
	if max_eigen is None or max_eigen[0] <= np.finfo(np.single).eps:
		# A nonpositive maximal eigenvalue indicates an indivisible network
		return cs.CommunitySet(G, np.zeros(G.max_node_idx + 1, np.int), 1)

	cdef np.ndarray community_vector = compute_community_vector(max_eigen[1])
	cdef float mod = modularity(mod_mtx, community_vector, G.size())
	#If modularity is not positive, network is indivisible
	if mod <= np.finfo(np.single).eps:
		return cs.CommunitySet(G, np.zeros(G.max_node_idx + 1, np.int), 1)

	if fine_tune:
		fine_tune_modularity(mod_mtx, community_vector, mod, G.size())

	comm0 = []
	comm1 = []

	# Create lists of indices in both communities
	for i in range(len(community_vector)):
		if community_vector[i] == -1:
			community_vector[i] = 0
			comm0.append(i)
		else:
			comm1.append(i)

	subdivision_queue = deque([comm0, comm1])
	cdef max_cidx = 1
	while len(subdivision_queue) > 0:
		# Repeat the above procedure repeatedly on each new community.
		# The main difference is that here we use a smaller modularity matrix,
		# defined only on the group at hand.
		members = subdivision_queue.popleft()
		
		mod_mtx_g = group_modularity_matrix(mod_mtx, members)

		max_eigen = get_max_eigen(mod_mtx_g)
		if max_eigen is None or max_eigen[0] <= np.finfo(np.single).eps:
			continue;

		subcommunity_vec = compute_community_vector(max_eigen[1])
		mod_g = modularity(mod_mtx_g, subcommunity_vec, G.size())
		if mod_g <= np.finfo(np.single).eps:
			continue;

		if fine_tune:
			fine_tune_modularity(mod_mtx_g, subcommunity_vec, mod_g, G.size())

		new_comm_0 = []
		new_comm_1 = []

		for i in range(len(subcommunity_vec)):
			if subcommunity_vec[i] == -1:
				community_vector[members[i]] = max_cidx + 1
				new_comm_0.append(members[i])
			else:
				community_vector[members[i]] = max_cidx + 2
				new_comm_1.append(members[i])
		
		subdivision_queue.append(new_comm_0)
		subdivision_queue.append(new_comm_1)
		max_cidx += 2

	num_communities = common.normalize_communities(community_vector)
	return cs.CommunitySet(G, community_vector, num_communities)
