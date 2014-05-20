from zen import Graph, ZenException
from collections import deque
import communityset as cs
import numpy as np

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
	comm_vec = np.empty(eigenvec.shape[0])

	for i in range(eigenvec.shape[0]):
		if(eigenvec[i] > 0):
			comm_vec[i] = 1
		else:
			comm_vec[i] = 0

	return comm_vec

# Compute the modularity of the graph with the given community vector.
# Return True if the modularity is non-negative, False otherwise.
def check_modularity_sign(mod_matrix, comm_vector):
	q = np.dot(comm_vector, np.dot(mod_matrix, comm_vector))
	return (q > 0)

# Modifies the adjacency matrix to make it a modularity matrix.
def as_modularity_matrix(adj_mat, G):
	for i in range(adj_mat.shape[0]):
		for j in range(adj_mat.shape[1]):
			factor = (G.degree_(i) * G.degree_(j)) / (2.0 * G.size())
			adj_mat[i,j] -= factor

# Compute a modularity matrix for a part of the graph
def group_modularity_matrix(mod, group):
	mod_g = np.empty((len(group), len(group)))
	
	for i in range(len(group)):
		for j in range(len(group)):
			mod_g[i,j] = mod[group[i],group[j]]

	for i in range(len(group)):
		mod_g[i,i] -= np.sum(mod_g[:,i])

	return mod_g

def spectral_modularity(G):

	if G.is_directed():
		raise ZenException("This algorithm only supports undirected graphs.")
	if not G.is_compact():
		raise ZenException("The graph must be compact.")
	# TODO This does not support weighted networks, prevent it

	mod = G.matrix()
	as_modularity_matrix(mod, G)

	max_eigen = get_max_eigen(mod)
	if max_eigen is None or max_eigen[0] <= 0:
		# A nonpositive maximal eigenvalue indicates an indivisible network
		return cs.CommunitySet(G, np.zeros(G.max_node_index + 1))

	community_vector = compute_community_vector(max_eigen[1])
	#If modularity is not positive, network is indivisible
	if not check_modularity_sign(mod, community_vector):
		return cs.CommunitySet(G, np.zeros(G.max_node_index + 1))

	comm0 = []
	comm1 = []

	for i in range(len(community_vector)):
		if community_vector[i] == 0:
			comm0.append(i)
		else:
			comm1.append(i)

	subdivision_queue = deque([comm0, comm1])
	max_cidx = 1
	while len(subdivision_queue) > 0:
		members = subdivision_queue.popleft()
		
		mod_g = group_modularity_matrix(mod, members)

		max_eigen = get_max_eigen(mod_g)
		if max_eigen is None or max_eigen[0] <= 0:
			continue;

		subcommunity_vector = compute_community_vector(max_eigen[1])
		if not check_modularity_sign(mod_g, subcommunity_vector):
			continue;

		new_comm_0 = []
		new_comm_1 = []

		for i in range(len(subcommunity_vector)):
			if subcommunity_vector[i] == 0:
				community_vector[members[i]] = max_cidx + 1
				new_comm_0.append(members[i])
			else:
				community_vector[members[i]] = max_cidx + 2
				new_comm_1.append(members[i])
		
		subdivision_queue.append(new_comm_0)
		subdivision_queue.append(new_comm_1)
		max_cidx += 2

	return cs.CommunitySet(G, community_vector)
