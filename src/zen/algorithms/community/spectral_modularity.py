from zen import Graph, ZenException
from collections import deque
import communityset as cs
import numpy as np
import community_common as common

# Obtain the max eigenvalue and its associated eigenvector
# from a real symmetric matrix
def get_max_eigen(mat):
	vals, vects = np.linalg.eigh(mat)

	if len(vals) == 0:
		return None

	max_i = 0

	for i in range(1, len(vals)):
		if vals[i] > vals[max_i]:
			max_i = i
			
	return (vals[max_i], vects[:, max_i])

# Create a community vector from an eigenvector. Based on the eigenvector
# signs, the community vector entries are either 1 (first community) or -1
# (second community)
def compute_community_vector(eigenvec):
	comm_vec = np.empty(eigenvec.shape[0], np.int)

	for i in range(eigenvec.shape[0]):
		if(eigenvec[i] > 0):
			comm_vec[i] = 1
		else:
			comm_vec[i] = -1

	return comm_vec

def modularity(mod_mtx, comm_vec, num_edges):
	return (1.0 / (4 * num_edges)) * np.dot(comm_vec, np.dot(mod_mtx, comm_vec))

def fine_tune_modularity(mod_matrix, comm_vector, initial_mod, num_edges):
	prev_mod = initial_mod

	while True:
		next_comm_vector = comm_vector.copy()
		to_move = set(range(len(comm_vector)))
		current_mod = prev_mod
		movements = []

		while len(to_move) > 0:
			max_mod_delta = -2
			max_idx = -1
	
			for i in to_move:
				# Move to the other community, compute modularity			
				next_comm_vector[i] *= -1
				mod_delta = (modularity(mod_matrix, next_comm_vector, num_edges)
							- prev_mod)
	
				# Record if we have better modularity with this node 
				if(mod_delta >= max_mod_delta):
					max_idx = i
					max_mod_delta = mod_delta

				# Move back to test the next node
				next_comm_vector[i] *= -1
		
			# Move the node that gives the biggest increase / smallest decrease
			to_move.remove(max_idx)
			next_comm_vector[max_idx] *= 1

			# Record this in the movement list
			current_mod += max_mod_delta
			movements.append((max_idx, current_mod))

		# Now that all nodes have been moved, select the state which had the
		# best modularity.
		best_mod_idx = 0
		for i in range(1, len(movements)):
			if(movements[i][1] > movements[best_mod_idx][1]):
				best_mod_idx = i

		# If we could not improve our overall modularity, we're done
		if movements[best_mod_idx][1] < prev_mod:
			break
		
		# Otherwise replicate all the given movements on the community vector
		prev_mod = movements[best_mod_idx][1]
		for i in range(best_mod_idx + 1):
			comm_vector[movements[i][0]] *= -1

	#Nothing to return, we modified the community vector in-place

# Modifies the adjacency matrix to make it a modularity matrix.
def as_modularity_matrix(adj_mat, G):
	for i in range(adj_mat.shape[0]):
		for j in range(adj_mat.shape[1]):
			factor = (G.degree_(i) * G.degree_(j)) / (2.0 * G.size())
			adj_mat[i,j] -= factor

# Compute a modularity matrix for a part of the graph
def group_modularity_matrix(mod, group):
	mod_mtx_g = np.empty((len(group), len(group)))
	
	for i in range(len(group)):
		for j in range(len(group)):
			mod_mtx_g[i,j] = mod[group[i],group[j]]

	for i in range(len(group)):
		mod_mtx_g[i,i] -= np.sum(mod_mtx_g[:,i])

	return mod_mtx_g

def spectral_modularity(G, **kwargs):
	"""
	Detect communities in a graph using the method described in [Newman, 2006].
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
        
    ..seealso::
        Newman 2006...
        
	"""
	if G.is_directed():
		raise ZenException("This algorithm only supports undirected graphs.")
	if not G.is_compact():
		raise ZenException("The graph must be compact.")

	edges = G.edges_(weight=True)
	for i in range(len(edges)):
		if edges[i,1] != 1:
			raise ZenException("This algorithm only supports unweighted graphs.")

	fine_tune = kwargs.pop("fine_tune", False);

	# Empty graph: no communities
	if len(G) == 0:
		return cs.CommunitySet(G, [], [])

	mod_mtx = G.matrix()
	as_modularity_matrix(mod_mtx, G)

	max_eigen = get_max_eigen(mod_mtx)
	if max_eigen is None or max_eigen[0] <= np.finfo(np.single).eps:
		# A nonpositive maximal eigenvalue indicates an indivisible network
		num_elems = G.max_node_idx + 1
		return cs.CommunitySet(G, np.zeros(num_elems, np.int), [num_elems])

	community_vector = compute_community_vector(max_eigen[1])
	mod = modularity(mod_mtx, community_vector, G.size())
	#If modularity is not positive, network is indivisible
	if mod <= np.finfo(np.single).eps:
		num_elems = G.max_node_idx + 1
		return cs.CommunitySet(G, np.zeros(num_elems, np.int), [num_elems])

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
	max_cidx = 1
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

	community_sizes = common.normalize_communities(community_vector)
	return cs.CommunitySet(G, community_vector, community_sizes)
