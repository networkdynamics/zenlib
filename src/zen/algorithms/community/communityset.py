from zen import Graph, ZenException
import numpy as np

## Enforce policy that all community numbers are continguous from [0-M].

class Community:
	"""
	This class represents a community that was discovered in a :py:class:`Graph` 
	using one of the community detection algorithms. This class should not be
	instantiated directly but rather obtained from a :py:class:`CommunitySet`.

	Public properties:
	
		* ``nodes``: a set containing the indices of the community nodes.
		* ``community_idx``: the index of this community in the 
		:py:class:`CommunitySet` it originates from.
	"""
    ## TODO: Consider changing list to a set - faster for lookup
    
	def __init__(self, idx, G, node_list):
		self._graph = G
		self.nodes = np.sort(node_list)
		self.community_idx = idx

	def	has_node_index(self, nidx):
		"""
		Return ``True`` if this community contains the node with node index
		``nidx`` 
		"""
		# searchsorted returns the index where we should insert the value nidx. 
		# So if nidx is already present in the array, idx will be its index.
		idx = np.searchsorted(self.nodes, nidx)
		return self.nodes[idx] == nidx

	def __contains__(self, nobj):
		"""
		Return ``True`` if this community contains the node with node object
		``nobj``. If ``nobj`` does not belong to the graph associated to this
		community, return ``False``.
		"""
		try:
			nidx = self._graph.node_idx(nobj)
		except:
			return False
		return self.has_node_index(nidx)

class CommunitySet:
	"""
	A set of non-overlapping communities detected in a :py:class:`Graph`. This 
	class should not be instantiated directly but rather obtained from the
	algorithms in the `zen.algorithms.community` package.
	"""
	def __init__(self, G, communities, community_sizes):
		self._graph = G

		# Array of communities, indexed by node indices (i.e. A[i] is the
		# community index of node with index i in graph G).
		self._communities = communities

		# Array of community sizes, indexed by community indices
		self._community_sizes = community_sizes

	def __raise_if_invalid_nidx(self, nidx):
        ## TODO: This is a very expensive way of checking validity. Maybe add a method to Graph/DiGraph
		if self._graph.node_object(nidx) is None:
			raise ZenException, 'Invalid node idx %d' % nidx

	def __build_community(self, cidx):
		c_nodes = np.empty(self._community_sizes[cidx])
		i = 0
		for nidx in self._graph.nodes_():
			if self._communities[nidx] == cidx:
				c_nodes[i] = nidx
				i += 1

		return Community(cidx, self._graph, np.array(c_nodes))

	def __build_all_communities(self):
		communities = [[] for i in range(len(self))]

		for nidx in self._graph.nodes_():
			cidx = self._communities[nidx]
			communities[cidx].append(nidx)

		
		for i in range(len(self)):
			communities[i] = Community(i, self._graph, communities[i])

		return communities

	def __len__(self):
		"""
		Returns the number of communities in this set
		"""
		return len(self._community_sizes)

	def communities(self):
		"""
		Return a list of the communities contained in this community set.
		"""
		return self.__build_all_communities()

	def __iter__(self):
		"""
		Iterate through communities in this set.
		"""
		for cidx in range(len(self)):
			yield __build_community(cidx)

	def community_(self, nidx):
		"""
		Return the community associated with node having index ``nidx``.
		"""
		self.__raise_if_invalid_nidx(nidx)
		return self.__build_community(self._communities[nidx])

	def community(self, nobj):
		"""
		Return the community associated with the node having object identifier
		``nobj``.
		"""
		nidx = self._graph.node_idx(nobj)
		return self.community_(nidx)

	def community_idx_(self, nidx):
		"""
		Return the index of the community associated with node having index 
		``nidx``.
		"""
		self.__raise_if_invalid_nidx(nidx)
		return self._communities[nidx]	

	def community_idx(self, nobj):
		"""
		Return the community index associated with the node having object 
		identifier ``nobj``.
		"""
		nidx = self._graph.node_idx(nobj)
		return self.community_idx_(nidx)

	def share_community_(self, u_idx, v_idx):
		"""
		Return ``True`` if nodes u and v (node indices) are in the same 
		community.
		"""
		return self.community_idx_(u_idx) == self.community_idx_(v_idx)

	def share_community(self, u_obj, v_obj):
		"""
		Return ``True`` if nodes u and v (node objects) are in the same 
		community.
		"""
		try:
			u_idx = self._graph.node_idx(u_obj)
			v_idx = self._graph.node_idx(v_obj)
		except:
			return False
		return self.community_idx(u_idx) == self.community_idx(v_idx)

	
