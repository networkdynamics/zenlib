from zen import Graph, ZenException
import numpy as np

## TODO: Add significant documentation here
## Enforce policy that all community numbers are continguous from [0-M].

class Community:
    ## TODO: Consider changing list to a set - faster for lookup
    
	def __init__(self, idx, G, node_list):
		self._graph = G
		self.nodes = np.sort(node_list)
		self.community_idx = idx

	def	has_node_(self, nidx):
		"""
		Return ``True`` if this community contains the node with node index
		``nidx`` 
		"""
		# searchsorted returns the index where we should insert the value nidx. 
		# So if nidx is already present in the array, idx will be its index.
		idx = np.searchsorted(self.nodes, nidx)
		return self.nodes[idx] == nidx

    ## TODO: __contains__ is a pythonic way of handling "has"
	def has_node(self, nobj):
		"""
		Return ``True`` if this community contains the node with node object
		``nobj``. If ``nobj`` does not belong to the graph associated to this
		community, return ``False``
		"""
		try:
			nidx = self._graph.node_idx(nobj)
		except:
			return False
		return self.has_node_(nidx) 

class CommunitySet:
	# TODO Various methods of this class (those that build communities, mainly)
	# are implemented without regard to performance. This should eventually
	# be fixed.

	# TODO Implement __len__?

	def __init__(self, G, communities):
		self._graph = G
		self._communities = communities

	def __raise_if_invalid_nidx(self, nidx):
        ## TODO: This is a very expensive way of checking validity. Maybe add a method to Graph/DiGraph
		if self._graph.node_object(nidx) is None:
			raise ZenException, 'Invalid node idx %d' % nidx

	def __build_community(self, cidx):
		c_nodes = []
		for nidx in self._graph.nodes_():
			if self._communities[nidx] == cidx:
				c_nodes.append(nidx)

		return Community(cidx, self._graph, np.array(c_nodes))

	def __build_all_communities(self):
		communities = {}
		for nidx in self._graph.nodes_():
			cidx = self._communities[nidx]
			if cidx not in communities:
				communities[cidx] = [nidx]
			else:
				communities[cidx].append(nidx)

		community_list = []
		for cidx, nodes in communities.iteritems():
			community_list.append(Community(cidx, self._graph, nodes))

		return community_list

	def communities(self):
		"""
		Return a list of the communities contained in this community set.
		"""
		return self.__build_all_communities()

	def __iter__(self):
		built_communities = set()
		for nidx in self._graph.nodes_():
			cidx = self._communities[nidx]
			if cidx not in built_communities:
				built_communities.add(cidx)
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
		Return ``True`` if nodes u and v are in the same community.
		"""
		return self.community_idx_(u_idx) == self.community_idx_(v_idx)

	def share_community(self, u_obj, v_obj):
		"""
		Return ``True`` if nodes u and v are in the same community.
		"""
		try:
			u_idx = self._graph.node_idx(u_obj)
			v_idx = self._graph.node_idx(v_obj)
		except:
			return False
		return self.community_idx(u_idx) == self.community_idx(v_idx)

	
