from zen.graph cimport Graph
from zen.digraph cimport DiGraph
from zen import ZenException

cimport numpy as np

from cpython cimport bool

cdef class Community:
	"""
	This class represents a community that was discovered in a :py:class:`Graph` 
	using one of the community detection algorithms. This class should not be
	instantiated directly but rather obtained from a :py:class:`CommunitySet`.

	Public properties:
	
		* ``community_idx``: the index of this community in the 
		:py:class:`CommunitySet` it originates from.
	"""

	def __init__(Community self, int idx, G, set node_set, dict probs=None):

		if type(G) != Graph and type(G) != DiGraph:
			raise ZenException, 'Unknown graph type: %s' % type(G)

		self._graph = G
		self._nodes = node_set
		self.community_idx = idx
		self._probabilities = probs

	def __len__(Community self):
		return len(self._nodes)

	def __contains__(Community self, nobj):
		"""
		Return ``True`` if this community contains the node with node object
		``nobj``. If ``nobj`` does not belong to the graph associated to this
		community, return ``False``.
		"""
		cdef int nidx
		try:
			nidx = self._graph.node_idx(nobj)
		except KeyError:
			return False
		return self.has_node_index(nidx)

	def __iter__(Community self):
		"""
		Iterate over the nodes of this community

		TODO: This should iterate over node objects
		"""
		return iter(self._nodes)

	def iter_(Community self):
		"""
		TODO: This should iterate over node indexes
		"""
		raise NotImplemented

	def assoc_prob_(Community self, int nidx):
		"""
		Returns the probability that the given node index belongs to this
		community, as given by the community detection algorithm.
		"""
		if self._probabilities is None:
			return 1.0 if self.has_node_index(nidx) else 0.0
		
		return self._probabilities[nidx] if nidx in self._probabilities else 0.0
	
	def assoc_prob(Community self, nobj):
		"""
		Returns the probability that the given node object belongs to this
		community, as given by the community detection algorithm. If the node
		object does not belong to the graph, 0 will be returned.
		"""
		cdef int nidx
		try:
			nidx = self._graph.node_idx(nobj)
		except KeyError:
			return 0.0
		return self.assoc_prob_(nidx)

	cpdef bool has_node_index(Community self, int nidx):
		"""
		Return ``True`` if this community contains the node with node index
		``nidx`` 
		"""
		return nidx in self._nodes

cdef class CommunitySet:
	"""
	A set of non-overlapping communities detected in a :py:class:`Graph`. This 
	class should not be instantiated directly but rather obtained from the
	algorithms in the `zen.algorithms.community` package.
	"""
	def __init__(CommunitySet self, G, np.ndarray communities, 
					int num_communities):

		if type(G) != Graph and type(G) != DiGraph:
			raise ZenException, 'Unknown graph type: %s' % type(G)

		self._graph = G
		self._num_communities = num_communities

		# Array of communities, indexed by node indices (i.e. A[i] is the
		# community index of node with index i in graph G).
		self._communities = communities

	cdef void __raise_if_invalid_nidx(CommunitySet self, int nidx) except *:
		if not self._graph.is_valid_node_idx(nidx):
			raise ZenException, 'Invalid node idx %d' % nidx

	cdef Community __build_community(CommunitySet self, int cidx):
		cdef set nodes = set()
		cdef int nidx
		for nidx in self._graph.nodes_():
			if self._communities[nidx] == cidx:
				nodes.add(nidx)

		return Community(cidx, self._graph, nodes)

	cdef list __build_all_communities(CommunitySet self):
		cdef:
			int nidx
			int cidx
			int i
		communities = [set() for i in range(len(self))]
		
		for nidx in self._graph.nodes_():
			cidx = self._communities[nidx]
			communities[cidx].add(nidx)

		
		for i in range(len(self)):
			communities[i] = Community(i, self._graph, communities[i])

		return communities

	def __len__(CommunitySet self):
		"""
		Returns the number of communities in this set
		"""
		return self._num_communities

	def communities(CommunitySet self):
		"""
		Return a list of the communities contained in this community set.

		TODO: This should only build the communities on the first call.  Cached otherwise.
		"""
		return self.__build_all_communities()

	def community(CommunitySet self, int cidx):
		"""
		Return the community with index ``cidx``

		TODO: Build the community only once (and don't build all other communities).  Cached otherwise.
		"""
		return self.__build_community(cidx)

	def __iter__(CommunitySet self):
		"""
		Iterate through communities in this set.

		TODO: See self.communities() method changes.  Cache when possible.
		"""
		cdef int cidx
		for cidx in range(len(self)):
			yield self.__build_community(cidx)

	cpdef list node_communities_(CommunitySet self, int nidx):
		"""
		Return the community associated with node having index ``nidx``.

		TODO: Cache when possible.
		"""
		self.__raise_if_invalid_nidx(nidx)
		return [self.__build_community(self._communities[nidx])]

	def node_communities(CommunitySet self, nobj):
		"""
		Return the community associated with the node having object identifier
		``nobj``.
		"""
		cdef int nidx = self._graph.node_idx(nobj)
		return self.node_communities_(nidx)

	"""
	TODO: Implement node_community, node_community_, node_community_index, and node_community_index_
	"""

	cpdef list node_community_indices_(CommunitySet self, int nidx):
		"""
		Return the index of the community associated with node having index 
		``nidx``.
		"""
		self.__raise_if_invalid_nidx(nidx)
		return [self._communities[nidx]]

	def node_community_indices(CommunitySet self, nobj):
		"""
		Return the community index associated with the node having object 
		identifier ``nobj``.
		"""
		cdef int nidx = self._graph.node_idx(nobj)
		return self.node_community_indices_(nidx)

	cpdef bool share_community_(CommunitySet self, int u_idx, int v_idx):
		"""
		Return ``True`` if nodes u and v (node indices) are in the same 
		community.

		TODO: check_sharing_
		"""
		self.__raise_if_invalid_nidx(u_idx)
		self.__raise_if_invalid_nidx(v_idx)

		# Written in this way to prevent a cast error between np.bool and bool
		if self.node_community_indices_(u_idx)[0] == self.node_community_indices_(v_idx)[0]:
			return True
		return False

	def share_community(CommunitySet self, u_obj, v_obj):
		"""
		Return ``True`` if nodes u and v (node objects) are in the same 
		community.

		TODO: Rename to check_sharing
		"""
		cdef int u_idx
		cdef int v_idx
		try:
			u_idx = self._graph.node_idx(u_obj)
			v_idx = self._graph.node_idx(v_obj)
		except:
			return False
		return self.share_community_(u_idx, v_idx)

	def __str__(CommunitySet self):
		return self._communities.__str__()
	
