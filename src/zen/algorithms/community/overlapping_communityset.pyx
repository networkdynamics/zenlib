from zen.graph cimport Graph
from zen.digraph cimport DiGraph

from communityset cimport Community
from zen import ZenException

cimport numpy as np

from cpython cimport bool

cdef class OverlappingCommunitySet:
	"""
	A set of overlapping communities detected in a :py:class:`Graph`. This 
	class should not be instantiated directly but rather obtained from the
	algorithms in the `zen.algorithms.community` package.
	"""
	def __init__(OverlappingCommunitySet self, G, np.ndarray communities, 
					int num_communities, dict probs=None):

		if type(G) != Graph and type(G) != DiGraph:
			raise ZenException, 'Unknown graph type: %s' % type(G)

		self._graph = G
		self._num_communities = num_communities

		# A[i] is the set of node indices in the community with index i
		self._communities = communities

		# A[(i, j)] is the probability of node i being in community j
		self._probabilities = probs

	cdef void __raise_if_invalid_nidx(OverlappingCommunitySet self, int nidx) except *:
		if not self._graph.is_valid_node_idx(nidx):
			raise ZenException, 'Invalid node idx %d' % nidx

	def __len__(OverlappingCommunitySet self):
		"""
		Returns the number of communities in this set
		"""
		return self._num_communities

	cdef __build_community(OverlappingCommunitySet self, int cidx):
		nodes = self._communities[cidx]

		if self._probabilities is None:
			return Community(cidx, self._graph, nodes)
		
		probs = {n: self._probabilities[(n, cidx)] for n in nodes}
		return Community(cidx, self._graph, nodes, probs)

	def communities(OverlappingCommunitySet self):
		"""
		Return a list of the communities contained in this community set.
		"""
		lst = []
		for cidx in range(len(self)):
			lst.append(self.__build_community(cidx))
		return lst

	def __iter__(OverlappingCommunitySet self):
		"""
		Iterate through communities in this set.
		"""
		for cidx in range(len(self)):
			yield self.__build_community(cidx)

	cpdef list node_communities_(OverlappingCommunitySet self, int nidx):
		"""
		Return the communities associated with node having index ``nidx``.
		"""
		self.__raise_if_invalid_nidx(nidx)
		comms = []
		for cidx, nodes in enumerate(self._communities):
			if nidx in nodes:
				comms.append(self.__build_community(cidx))
		return comms

	def node_communities(OverlappingCommunitySet self, nobj):
		"""
		Return the communities associated with the node having object identifier
		``nobj``.
		"""
		cdef int nidx = self._graph.node_idx(nobj)
		return self.node_communities_(nidx)

	cpdef list node_community_indices_(OverlappingCommunitySet self, int nidx):
		"""
		Return the index of the communities associated with node having index 
		``nidx``.
		"""
		self.__raise_if_invalid_nidx(nidx)
		return [i for i, nodes in enumerate(self._communities) if nidx in nodes]

	def node_community_indices(OverlappingCommunitySet self, nobj):
		"""
		Return the community indices associated with the node having object 
		identifier ``nobj``.
		"""
		cdef int nidx = self._graph.node_idx(nobj)
		return self.node_community_indices_(nidx)

	cpdef bool share_community_(OverlappingCommunitySet self, int u_idx, int v_idx):
		"""
		Return ``True`` if nodes u and v (node indices) share a community
		"""
		self.__raise_if_invalid_nidx(u_idx)
		self.__raise_if_invalid_nidx(v_idx)
		for nodes in self._communities:
			if u_idx in nodes and v_idx in nodes:
				return True
		return False

	def share_community(OverlappingCommunitySet self, u_obj, v_obj):
		"""
		Return ``True`` if nodes u and v (node objects) share a community
		"""
		cdef int u_idx
		cdef int v_idx
		try:
			u_idx = self._graph.node_idx(u_obj)
			v_idx = self._graph.node_idx(v_obj)
		except:
			return False
		return self.share_community_(u_idx, v_idx)
	
