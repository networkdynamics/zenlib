import numpy
import numpy as np
cimport numpy as np

import ctypes
from exceptions import *
from graph cimport Graph

from constants import AVG_OF_WEIGHTS, MAX_OF_WEIGHTS, MIN_OF_WEIGHTS, NO_NONE_LIST_OF_DATA, LIST_OF_DATA

cimport libc.stdlib as stdlib

cdef extern from "math.h" nogil:
	double fmax(double x, double y)
	double fmin(double x, double y)

# some things not defined in the cython stdlib header
cdef extern from "stdlib.h" nogil:
	void* memmove(void* destination, void* source, size_t size)

cdef extern from "math.h":
	double ceil(double x)
	double floor(double x)

__all__ = ['DiGraph','AVG_OF_WEIGHTS','MAX_OF_WEIGHTS','MIN_OF_WEIGHTS','NO_NONE_LIST_OF_DATA','LIST_OF_DATA']

cdef inline int imax(int a, int b):
	if a < b:
		return b
	else:
		return a

cdef inline int imin(int a, int b):
	if a < b:
		return a
	else:
		return b

"""
This data structure contains node info (in C-struct format) for fast array-based lookup.

When a node info entry is part of the free node list, the indegree points to the next
entry in the list and the in_capacity points to the previous entry in the list.
"""
cdef struct NodeInfo:
	# This data structure contains node info (in C-struct format) for fast array-based lookup.
	bint exists

	int indegree # The number of entries in the inelist that are in use
	int* inelist
	int in_capacity  # The length of the inelist

	int outdegree # the number of entries in outelist that are in use
	int* outelist
	int out_capacity # the length of outelist

cdef int sizeof_NodeInfo = sizeof(NodeInfo)

"""
When an edge info entry is part of the free edge list, src points to the next entry
in the list and tgt points to the previous entry.
"""
cdef struct EdgeInfo:
	bint exists
	int src # Node idx for the source of the edge
	int tgt # Node idx for the tgt of the edge
	double weight

cdef int sizeof_EdgeInfo = sizeof(EdgeInfo)

"""
The DiGraph class supports pickling.  Here we keep a record of changes from one pickle version to another.

Version 1.0:
	- The initial version (no diff)
"""
CURRENT_PICKLE_VERSION = 1.0

# static cython methods
cpdef DiGraph DiGraph_from_adj_matrix_np_ndarray(np.ndarray[np.float_t, ndim=2] M,
												node_obj_fxn):
	cdef DiGraph G = DiGraph()
	cdef int i
	cdef int rows = M.shape[0]
	cdef int cols = M.shape[1]

	# add nodes
	G.add_nodes(rows,node_obj_fxn)

	# add edges
	for i in range(rows):
		for j in range(cols):
			if M[i,j] != 0:
				G.add_edge_(i,j,None,M[i,j])

	return G


cdef class DiGraph:
	"""
	This class provides a highly-optimized implementation of a `directed graph <http://en.wikipedia.org/wiki/Directed_graph>`_.  Duplicate edges are not allowed.

	Public properties include:

		* ``max_node_index`` (int): the largest node index currently in use
		* ``max_edge_index`` (int): the largest edge index currently in use
		* ``edge_list_capacity`` (int): the initial number of edge positions that will be allocated in a newly created node's in and out edge lists.
		* ``node_grow_factor`` (int): the multiple by which the node storage array will grow when its capacity is exceeded.
		* ``edge_grow_factor`` (int): the multiple by which the edge storage array will grow when its capacity is exceeded.
		* ``edge_list_grow_factor`` (int): the multiple by which the a node's in/out edge list storage array will grow when its capacity is exceeded.

	**Graph Listeners**:

	Instances of a graph can notify one or more listeners of changes to it.  Listeners should support the following methods:

		* ``node_added(nidx,nobj,data)``
		* ``node_removed(nidx,nobj)``
		* ``edge_added(eidx,uidx,vidx,data,weight)``
		* ``edge_removed(eidx,uidx,vidx)``

	Other event notifications are possible (changes to data, etc...).  These will be supported in future versions.

	It is noteworthy that adding listeners imposes a serious speed limitation on graph building functions.  If no listeners
	are present in the graph, then node/edge addition/removal proceed as fast as possible.  Notifying listeners requires 
	these functions to follow non-optimal code paths.
	"""

	@staticmethod
	def from_adj_matrix(M,**kwargs):
		"""
		Create a new :py:class:`DiGraph` from adjacency matrix information
		contained in ``M``.  ``M`` can be a ``numpy.matrix`` or
		``numpy.ndarray`` with 2 dimensions.

		**Keyword Args**:

		  * ``node_obj_fxn [=int]`` (python function): The function that will be used to create
		    node objects from the node indices.  If ``None``, then no node objects will be created.
	
		"""
		# parse arguments
		node_obj_fxn = kwargs.pop('node_obj_fxn',int)

		if len(kwargs) > 0:
			raise ValueError, 'Keyword arguments not supported: %s' % ','.join(kwargs.keys())

		if type(M) == numpy.ndarray:
			return DiGraph_from_adj_matrix_np_ndarray(<np.ndarray[np.float_t,ndim=2]> M,node_obj_fxn)
		else:
			raise TypeError, 'Objects of type %s cannot be handled as adjancency matrices' % type(M)

	def __init__(DiGraph self,**kwargs):
		"""
		Create a new :py:class:`DiGraph` object.
	
		**Keyword Args**:

		  * ``node_capacity [=100]`` (int): the initial number of nodes this graph has space to hold.
		  * ``edge_capacity [=100]`` (int): the initial number of edges this graph has space to hold.
		  * ``edge_list_capacity [=5]`` (int): the initial number of edges that each node is allocated space for initially.

		"""
		node_capacity = kwargs.pop('node_capacity',100)
		edge_capacity = kwargs.pop('edge_capacity',100)
		edge_list_capacity = kwargs.pop('edge_list_capacity',5)

		cdef int i

		self.first_free_node = -1
		self.first_free_edge = -1

		self.num_changes = 0
		self.node_grow_factor = 1.5
		self.edge_list_grow_factor = 1.5
		self.edge_grow_factor = 1.5

		self.num_nodes = 0
		self.node_capacity = node_capacity
		self.next_node_idx = 0
		self.max_node_idx = -1
		self.node_info = <NodeInfo*> stdlib.malloc(sizeof_NodeInfo*self.node_capacity)
		for i in range(self.node_capacity):
			self.node_info[i].exists = False
		self.node_obj_lookup = {}
		self.node_data_lookup = {}
		self.node_idx_lookup = {}

		self.num_edges = 0
		self.edge_capacity = edge_capacity
		self.next_edge_idx = 0
		self.max_edge_idx = -1
		self.edge_info = <EdgeInfo*> stdlib.malloc(sizeof_EdgeInfo*self.edge_capacity)
		for i in range(self.edge_capacity):
			self.edge_info[i].exists = False
		self.edge_data_lookup = {}

		self.edge_list_capacity = edge_list_capacity
		
		self.num_graph_listeners = 0
		self.graph_listeners = set()

	def __dealloc__(DiGraph self):
		cdef int i
		# deallocate all node data (node_info)
		for i in range(self.next_node_idx):
			if self.node_info[i].exists:
				stdlib.free(self.node_info[i].inelist)
				stdlib.free(self.node_info[i].outelist)
		stdlib.free(self.node_info)

		# deallocate all edge data (edge_info)
		stdlib.free(self.edge_info)

	def __reduce__(self):
		return (DiGraph,tuple(),self.__getstate__())

	def __getstate__(self):

		state = {'PICKLE_VERSION':1.0}

		# store global details
		state['num_changes'] = self.num_changes

		# store all node details
		state['num_nodes'] = self.num_nodes
		state['node_capacity'] = self.node_capacity
		state['node_grow_factor'] = self.node_grow_factor
		state['next_node_idx'] = self.next_node_idx
		state['first_free_node'] = self.first_free_node
		state['node_obj_lookup'] = self.node_obj_lookup
		state['node_data_lookup'] = self.node_data_lookup
		state['node_idx_lookup'] = self.node_idx_lookup
		state['edge_list_capacity'] = self.edge_list_capacity
		state['edge_list_grow_factor'] = self.edge_list_grow_factor

		# store node_info
		node_info = []
		for i in range(self.node_capacity):
			pickle_entry = None

			if self.node_info[i].exists is not 0:
				pickle_entry = (bool(self.node_info[i].exists),
								self.node_info[i].indegree,
								self.node_info[i].in_capacity,
								[self.node_info[i].inelist[j] for j in range(self.node_info[i].indegree)],
								self.node_info[i].outdegree,
								self.node_info[i].out_capacity,
								[self.node_info[i].outelist[j] for j in range(self.node_info[i].outdegree)] )
			else:
				pickle_entry = (bool(self.node_info[i].exists),
								self.node_info[i].indegree,
								self.node_info[i].in_capacity)

			node_info.append(pickle_entry)

		state['node_info'] = node_info

		# store all edge details
		state['num_edges'] = self.num_edges
		state['edge_capacity'] = self.edge_capacity
		state['edge_grow_factor'] = self.edge_grow_factor
		state['next_edge_idx'] = self.next_edge_idx
		state['first_free_edge'] = self.first_free_edge
		state['edge_data_lookup'] = self.edge_data_lookup

		# store edge_info
		edge_info = []
		for i in range(self.edge_capacity):
			edge_info.append( (	bool(self.edge_info[i].exists),
								self.edge_info[i].src,
								self.edge_info[i].tgt,
								self.edge_info[i].weight) )

		state['edge_info'] = edge_info

		return state

	def __setstate__(self,state):

		# restore global details
		self.num_changes = state['num_changes']

		# restore all node details
		self.num_nodes = state["num_nodes"]
		self.node_capacity = state['node_capacity']
		self.node_grow_factor = state['node_grow_factor']
		self.next_node_idx = state['next_node_idx']
		self.first_free_node = state['first_free_node']
		self.node_obj_lookup = state['node_obj_lookup']
		self.node_data_lookup = state['node_data_lookup']
		self.node_idx_lookup = state['node_idx_lookup']
		self.edge_list_capacity = state['edge_list_capacity']
		self.edge_list_grow_factor = state['edge_list_grow_factor']

		# restore node_info
		self.node_info = <NodeInfo*> stdlib.malloc(sizeof_NodeInfo*self.node_capacity)
		self.max_node_idx = -1
		for i,entry in enumerate(state['node_info']):

			if entry[0] is True:
				self.max_node_idx = i
				exists, indegree, in_capacity, inelist, outdegree, out_capacity, outelist = entry
				self.node_info[i].exists = exists
				self.node_info[i].indegree = indegree
				self.node_info[i].in_capacity = in_capacity

				self.node_info[i].inelist = <int*> stdlib.malloc(sizeof(int) * in_capacity)
				for j,eidx in enumerate(inelist):
					self.node_info[i].inelist[j] = eidx

				self.node_info[i].outdegree = outdegree
				self.node_info[i].out_capacity = out_capacity

				self.node_info[i].outelist = <int*> stdlib.malloc(sizeof(int) * out_capacity)
				for j,eidx in enumerate(outelist):
					self.node_info[i].outelist[j] = eidx

			else:
				exists, indegree, in_capacity = entry
				self.node_info[i].exists = exists
				self.node_info[i].indegree = indegree
				self.node_info[i].in_capacity = in_capacity

				self.node_info[i].outdegree = -1
				self.node_info[i].out_capacity = -1

		# restore all edge details
		self.num_edges = state['num_edges']
		self.edge_capacity = state['edge_capacity']
		self.edge_grow_factor = state['edge_grow_factor']
		self.next_edge_idx = state['next_edge_idx']
		self.first_free_edge = state['first_free_edge']
		self.edge_data_lookup = state['edge_data_lookup']

		# restore edge_info
		self.edge_info = <EdgeInfo*> stdlib.malloc(sizeof_EdgeInfo*self.edge_capacity)
		self.max_edge_idx = -1
		for i,entry in enumerate(state['edge_info']):
			exists,src,tgt,weight = entry

			if exists == 1:
				self.max_edge_idx = i

			self.edge_info[i].exists = exists
			self.edge_info[i].src = src
			self.edge_info[i].tgt = tgt
			self.edge_info[i].weight = weight

		return

	def validate(self,**kwargs):
		"""
		Checks whether the graph structure is valid.

		This method inspects various invariants and conditions that should be present
		in order for the graph structure to be correct.  If any conditions are
		broken, an exception will be raised immediately.

		**KwArgs**:

			* ``verbose [=False]`` (boolean): print debugging information out before each condition check

		**Raises**:
			``AssertionError``: if a condition isn't satisfied.

		"""
		verbose = kwargs.pop('verbose',False)

		self.inner_validate(verbose)

	cdef inner_validate(self,bint verbose):
		cdef int i,j

		# self.next_node_idx < self.node_capacity
		if verbose:
			print 'checking if self.next_node_idx < self.node_capacity'

		assert self.next_node_idx <= self.node_capacity, 'self.next_node_idx > self.node_capacity (%d,%d)' % (self.next_node_idx,self.node_capacity)

		# self.max_node_idx < self.next_node_idx
		if verbose:
			print 'checking if self.max_node_idx < self.next_node_idx'

		assert self.max_node_idx < self.next_node_idx, 'self.max_node_idx >= self.next_node_idx (%d,%d)' % (self.max_node_idx,self.next_node_idx)

		# there should be no other valid nodes beyond self.max_node_idx
		if verbose:
			print 'checking if no valid nodes exist beyond self.max_node_idx'

		for i in range(self.max_node_idx+1,self.node_capacity):
			assert not self.node_info[i].exists, 'self.node_info[%d] exists, but is beyond self.max_node_idx (= %d)' % (i,self.max_node_idx)

		# the node entry preceeding self.max_node_idx should exist
		if verbose:
			print 'checking that self.max_node_idx node exists'

		assert self.max_node_idx >= -1, 'self.max_node_idx is invalid (= %d)' % self.max_node_idx
		assert self.max_node_idx == -1 or self.node_info[self.max_node_idx].exists, 'The node entry at self.max_node_idx does not exist'

		# count the number of existing nodes
		if verbose:
			print 'counting the number of existing nodes'

		cdef int num_existing_nodes = 0

		for i in range(self.next_node_idx):
			if self.node_info[i].exists:
				num_existing_nodes += 1

		assert num_existing_nodes == self.num_nodes, '# of existing nodes (%d) != self.num_nodes (%d)' % (num_existing_nodes,self.num_nodes)

		# validate the structure of the free node list
		if verbose:
			print 'counting the number of free nodes'

		cdef int num_free_nodes = 0
		i = self.first_free_node
		j = -1
		while i != -1:
			assert 0 <= i < self.node_capacity, 'Free node %d has an invalid index (self.node_capacity = %d)' % (i,self.node_capacity)

			assert not self.node_info[i].exists, 'Free node %d has exists flag set to True' % i

			num_free_nodes += 1

			assert self.node_info[i].in_capacity == j, 'Free node %d points to the incorrect predecessor (%d correct, %d actual)' % (i,j,self.node_info[i].in_capacity)

			j = i
			i = self.node_info[i].indegree

		assert (num_free_nodes + num_existing_nodes) == self.next_node_idx, '(# free nodes) + (# existing nodes) != self.next_node_idx (%d + %d != %d)' % (num_free_nodes,num_existing_nodes,self.next_node_idx)

		#####
		# Check edges

		# self.next_edge_idx < self.edge_capacity
		if verbose:
			print 'checking if self.next_edge_idx < self.edge_capacity'

		assert self.next_edge_idx <= self.edge_capacity, 'self.next_edge_idx > self.edge_capacity (%d,%d)' % (self.next_edge_idx,self.edge_capacity)

		# self.max_edge_idx < self.next_edge_idx
		if verbose:
			print 'checking if self.max_edge_idx < self.next_edge_idx'

		assert self.max_edge_idx < self.next_edge_idx, 'self.max_edge_idx >= self.next_edge_idx (%d,%d)' % (self.max_edge_idx,self.next_edge_idx)

		# there should be no other valid edges beyond self.max_edge_idx
		if verbose:
			print 'checking if no valid edges exist beyond self.max_edge_idx'

		for i in range(self.max_edge_idx+1,self.edge_capacity):
			assert not self.edge_info[i].exists, 'self.edge_info[%d] exists, but is beyond self.max_edge_idx (= %d)' % (i,self.max_edge_idx)

		# the edge entry preceeding self.max_edge_idx should exist
		if verbose:
			print 'checking that self.max_edge_idx edge exists'

		assert self.max_edge_idx >= -1, 'self.max_edge_idx is invalid (= %d)' % self.max_edge_idx
		assert self.max_edge_idx == -1 or self.edge_info[self.max_edge_idx].exists, 'The edge entry at self.max_edge_idx does not exist'

		# count the number of existing edges
		if verbose:
			print 'counting the number of existing edges'

		cdef int num_existing_edges = 0

		for i in range(self.next_edge_idx):
			if self.edge_info[i].exists:
				num_existing_edges += 1

		assert num_existing_edges == self.num_edges, '# of existing edges (%d) != self.num_edges (%d)' % (num_existing_edges,self.num_edges)

		# validate the structure of the free edge list
		if verbose:
			print 'counting the number of free edges'

		cdef int num_free_edges = 0
		i = self.first_free_edge
		j = -1
		while i != -1:
			assert 0 <= i < self.edge_capacity, 'Free edge %d has an invalid index (self.edge_capacity = %d)' % (i,self.edge_capacity)

			assert not self.edge_info[i].exists, 'Free edge %d has exists flag set to True' % i

			num_free_edges += 1

			assert self.edge_info[i].tgt == j, 'Free edge %d points to the incorrect predecessor (%d correct, %d actual)' % (i,j,self.edge_info[i].tgt)

			j = i
			i = self.edge_info[i].src

		assert (num_free_edges + num_existing_edges) == self.next_edge_idx, '(# free edges) + (# existing edges) != self.next_edge_idx (%d + %d != %d)' % (num_free_edges,num_existing_edges,self.next_edge_idx)

	def __getattr__(self,name):
		# TODO: Make num_sources and num_sinks native properties of DiGraph.
		if name == 'num_sources':
			num_sources = 0
			# count number of sources
			for i in range(self.next_node_idx):
				if self.node_info[i].exists:
					if self.node_info[i].indegree == 0:
						num_sources += 1
			return num_sources
			
		elif name == 'num_sinks':
			num_sinks = 0
			# count number of sources
			for i in range(self.next_node_idx):
				if self.node_info[i].exists:
					if self.node_info[i].outdegree == 0:
						num_sinks += 1
			return num_sinks

	def add_listener(self,listener):
		"""
		Add a listener to the graph that will be notified of all changes to the graph.
		"""
		self.graph_listeners.add(listener)
		self.num_graph_listeners = len(self.graph_listeners)

	def rm_listener(self,listener):
		"""
		Remove a listener so it will no longer be updated with changes to the graph.
		"""
		self.graph_listeners.remove(listener)
		self.num_graph_listeners = len(self.graph_listeners)

	cpdef np.ndarray[np.double_t] matrix(self):
		"""
		Construct and return the adjacency matrix.

		**Returns**:
			``np.ndarray[double,ndims=2]``, ``M``.  The topology of the graph in adjacency matrix form where ``M[i,j]`` is the
			weight of the edge between nodes with index ``i`` and ``j``.  If there is no edge between ``i`` and ``j``, then
			``M[i,j] = 0``.

				When using this function, keep in mind that
				if the graph is not compact, there may be some node indices that don't correspond to valid nodes.
				In this case, the corresponding matrix elements are not valid.  For example::

					G = DiGraph()
					G.add_node('a') # this node has index 0
					G.add_node('b') # this node has index 1
					G.add_node('c') # this node has index 2

					G.rm_node('b') # after this point, index 1 doesn't correspond to a valid node

					M = G.matrix() # M is a 3x3 matrix

					V = M[1,:] # the values in V are meaningless because a node with index 1 doesn't exist

				This situation can be resolved by making a call to :py:meth:`Graph.compact` prior to calling this function::

					G = DiGraph()
					G.add_node('a') # this node has index 0
					G.add_node('b') # this node has index 1
					G.add_node('c') # this node has index 2

					G.rm_node('b') # after this point, index 1 doesn't correspond to a valid node

					G.compact()
					# In the call above, we've reassigned node indices so that there are no invalid nodes
					# this means that nodes 'a' and 'b' have been assigned indices 0 and 1.

					M = G.matrix() # M is a 2x2 matrix

					V = M[1,:] # this is valid now and corresponds to node G.node_object(1)

		"""
		cdef np.ndarray[np.double_t,ndim=2] A = np.zeros( (self.next_node_idx,self.next_node_idx), np.double)
		cdef int i, j, u, eidx, v
		cdef double w

		for u in range(self.next_node_idx):
			if self.node_info[u].exists:
				for j in range(self.node_info[u].outdegree):
					eidx = self.node_info[u].outelist[j]
					v = self.endpoint_(eidx,u)
					w = self.edge_info[eidx].weight
					A[u,v] = w

		return A

	cpdef copy(DiGraph self):
		"""
		Create a copy of this graph.

		.. note:: that node and edge indices are preserved in this copy.

		**Returns**:
			:py:class:`zen.DiGraph`. A new graph object that contains an independent copy of the connectivity of this graph.
			Node objects and node/edge data in the new graph reference the same objects as in the old graph.
		"""
		cdef DiGraph G = DiGraph()
		cdef int i,j,eidx,eidx2
		cdef double weight

		for i in range(self.next_node_idx):
			if self.node_info[i].exists:
				nobj = self.node_object(i)
				ndata = self.node_data_(i)

				G.add_node_x(i,G.edge_list_capacity,G.edge_list_capacity,nobj,ndata)

		for eidx in range(self.next_edge_idx):
			if self.edge_info[eidx].exists:
				i = self.edge_info[eidx].src
				j = self.edge_info[eidx].tgt
				edata = self.edge_data_(eidx)
				weight = self.weight_(eidx)

				G.add_edge_x(eidx,i,j,edata,weight)

		return G

	cpdef bint is_directed(DiGraph self):
		"""
		Return ``True`` if this graph is directed (which it is).
		"""
		return True

	cpdef bint is_compact(DiGraph self):
		"""
		Return ``True`` if the graph is in compact form.

		A graph is compact if there are no unallocated node or edge indices.
		The graph can be compacted by calling the :py:meth:`DiGraph.compact` method.
		"""
		return (self.num_nodes == (self.max_node_idx + 1) and self.num_edges == (self.max_edge_idx + 1))

	cpdef compact(DiGraph self):
		"""
		Compact the graph in place.  This will re-assign:

			#. node indices such that there are no unallocated node indices less than self.max_node_idx
			#. edge indices such that there are no unallocated edge indices less than self.max_edge_idx

		.. note:: At present no way is provided of keeping track of the changes made to node and edge indices.
		"""
		cdef int next_free_idx
		cdef int src, dest
		cdef int u,v,i
		cdef int nidx,eidx

		#####
		# move the nodes around
		nidx = 0
		while nidx < self.max_node_idx:

			if self.node_info[nidx].exists:
				nidx += 1
				continue

			# move all node content
			src = self.max_node_idx
			dest = nidx
			self.node_info[dest] = self.node_info[src]
			self.node_info[src].exists = False

			# move the node object references
			if src in self.node_obj_lookup:
				obj = self.node_obj_lookup[src]
				self.node_obj_lookup[dest] = obj
				del self.node_obj_lookup[src]
				self.node_idx_lookup[obj] = dest

			# move the node data
			if src in self.node_data_lookup:
				self.node_data_lookup[dest] = self.node_data_lookup[src]
				del self.node_data_lookup[src]

			#####
			# modify all the out-bound edges (and the relevant nodes)
			for i in range(self.node_info[dest].outdegree):
				eidx = self.node_info[dest].outelist[i]

				# get the other node whose edge list we need to update
				v = self.edge_info[eidx].tgt

				#####
				# update the other node's edge list

				# remove the entry for src
				self.__remove_edge_from_inelist(v,eidx,src)

				self.edge_info[eidx].src = dest

				# insert the entry for dest
				self.__insert_edge_into_inelist(v,eidx,dest)

			#####
			# modify all the in-bound edges (and the relevant nodes)
			for i in range(self.node_info[dest].indegree):
				eidx = self.node_info[dest].inelist[i]

				# get the other node whose edge list we need to update
				u = self.edge_info[eidx].src

				#####
				# update the other node's edge list

				# remove the entry for src
				self.__remove_edge_from_outelist(u,eidx,src)

				self.edge_info[eidx].tgt = dest

				# insert the entry for dest
				self.__insert_edge_into_outelist(u,eidx,dest)

			# update the max node index
			while self.max_node_idx >= 0 and not self.node_info[self.max_node_idx].exists:
				self.max_node_idx -= 1

			# move to the next node
			nidx += 1

		self.next_node_idx = self.max_node_idx + 1
		self.first_free_node = -1

		#####
		# move the edges around
		eidx = 0
		while eidx < self.max_edge_idx:
			if self.edge_info[eidx].exists:
				eidx += 1
				continue

			# move all edge content from the last element to the first free edge
			src = self.max_edge_idx
			dest = eidx
			self.edge_info[dest] = self.edge_info[src]
			self.edge_info[src].exists = False

			# move the edge data
			if src in self.edge_data_lookup:
				self.edge_data_lookup[dest] = self.edge_data_lookup[src]
				del self.edge_data_lookup[src]

			#####
			# change entries in u and v
			u = self.edge_info[dest].src
			v = self.edge_info[dest].tgt

			# in u
			i = self.find_outelist_insert_pos(self.node_info[u].outelist,self.node_info[u].outdegree,v)
			self.node_info[u].outelist[i] = dest

			# in v
			i = self.find_inelist_insert_pos(self.node_info[v].inelist,self.node_info[v].indegree,u)
			self.node_info[v].inelist[i] = dest

			# wipe out the source edge info
			self.edge_info[src].src = -1
			self.edge_info[src].tgt = -1

			# update the max node index
			while self.max_edge_idx >= 0 and not self.edge_info[self.max_edge_idx].exists:
				self.max_edge_idx -= 1

			# move to the next edge
			eidx += 1

		# at this point the max number of nodes and the next node available are adjacent
		self.first_free_edge = -1
		self.next_edge_idx = self.max_edge_idx + 1

		return

	cpdef skeleton(self,data_merge_fxn=NO_NONE_LIST_OF_DATA,weight_merge_fxn=AVG_OF_WEIGHTS):
		"""
		Create an undirected version of this graph.

		.. note::
			Node indices will be preserved in the undirected graph that is returned.  Edge indices,
			however, will not due to the fact that the number of edges present in the graph returned
			may be smaller.

		**Args**:

			* ``data_merge_fxn [=NO_NONE_LIST_OF_DATA]``: decides how the data objects associated
				with reciprocal	edges will be combined into a data object for a single undirected edge.
				Valid values values are:

				* ``NO_NONE_LIST_OF_DATA``: a list of the data objects if the both data
			  		objects are not None.  Otherwise, the value will be set to None.
				* ``LIST_OF_DATA``: a list of the data objects regardless of their values.
				* an arbitrary function of the form ``merge(i,j,d1,d2)`` that returns a single object.
			  		In this instance, ``d1`` is the data associated with edge ``(i,j)`` and
			  		``d2`` is the data object associated with edge ``(j,i)``.  Note that ``i`` and ``j``
			  		are node indices, not objects.

			* ``weight_merge_fxn [=AVG_OF_WEIGHTS]``: decides how the values of reciprocal edges will be
				combined into a single undirected edge.  Valid values are:

				* ``AVG_OF_WEIGHTS``: average the two weights
			 	* ``MIN_OF_WEIGHTS``: take the min value of the weights
				* ``MAX_OF_WEIGHTS``: take the max value of the two weights
				* an arbitrary function of the form ``merge(i,j,w1,w2)`` that returns a ``float``.
			  In this instance, ``w1`` is the weight associated with edge ``(i,j)`` and
			  ``w2`` is the weight associated with edge ``(j,i)``. Note that ``i`` and ``j`` are
			  node indices, not objects.

		**Returns**:
			:py:class:`zen.Graph`. The undirected version of this graph.

		**Raises**:
			:py:exc:`zen.ZenException`: if either ``data_merge_fxn`` or ``weight_merge_fxn`` are not properly defined.
		"""
		cdef Graph G = Graph()
		cdef int i,j,eidx,eidx2
		cdef double weight, weight2

		cdef int _CUSTOM_WEIGHT_FXN = -1
		cdef int _AVG_OF_WEIGHTS = AVG_OF_WEIGHTS
		cdef int _MIN_OF_WEIGHTS = MIN_OF_WEIGHTS
		cdef int _MAX_OF_WEIGHTS = MAX_OF_WEIGHTS

		cdef int _CUSTOM_DATA_FXN = -1
		cdef int _NO_NONE_LIST_OF_DATA = NO_NONE_LIST_OF_DATA
		cdef int _LIST_OF_DATA = LIST_OF_DATA

		cdef int weight_merge_switch = _CUSTOM_WEIGHT_FXN
		if type(weight_merge_fxn) == int:
			weight_merge_switch = weight_merge_fxn

		cdef int data_merge_switch = _CUSTOM_DATA_FXN
		if type(data_merge_fxn) == int:
			data_merge_switch = data_merge_fxn

		for i in range(self.next_node_idx):
			if self.node_info[i].exists:
				nobj = self.node_object(i)
				ndata = self.node_data_(i)

				# adding the node this way preserves the node indices
				G.add_node_x(i,G.edge_list_capacity,nobj,ndata)

		for eidx in range(self.next_edge_idx):
			if self.edge_info[eidx].exists:
				i = self.edge_info[eidx].src
				j = self.edge_info[eidx].tgt
				edata = self.edge_data_(eidx)
				weight = self.weight_(eidx)

				if not G.has_edge_(i,j):
					eidx2 = G.add_edge_(i,j,edata)
					G.set_weight_(eidx2,weight)
				else:
					eidx2 = G.edge_idx_(i,j)

					### Merge the weights
					weight2 = G.weight_(eidx2)

					if weight_merge_switch == _AVG_OF_WEIGHTS:
						weight2 = (weight2 + weight) / 2.0
					elif weight_merge_switch == _MIN_OF_WEIGHTS:
						weight2 = fmin(weight2,weight)
					elif weight_merge_switch == _MAX_OF_WEIGHTS:
						weight2 = fmax(weight2,weight)
					elif weight_merge_switch == _CUSTOM_WEIGHT_FXN:
						weight2 = weight_merge_fxn(i,j,weight,weight2)
					else:
						raise ZenException, 'Weight merge function switch %d (weight_merge_fxn = %s) is not defined' % (weight_merge_switch,str(weight_merge_fxn))

					G.set_weight_(eidx2,weight2)

					### Merge the data
					edata2 = G.edge_data_(eidx2)
					if data_merge_switch == _NO_NONE_LIST_OF_DATA and edata is None and edata2 is None:
						# nothing to do since we don't make lists from double None data objects
						pass
					elif data_merge_switch == _NO_NONE_LIST_OF_DATA or data_merge_switch == _LIST_OF_DATA:
						G.set_edge_data_(eidx2,[edata2,edata])
					elif data_merge_switch == _CUSTOM_DATA_FXN:
						G.set_edge_data_(eidx2,data_merge_fxn(i,j,edata,edata2))
					else:
						raise ZenException, 'Data merge function switch %d (data_merge_fxn = %s) is not defined' % (data_merge_switch,str(data_merge_fxn))

		return G

	cpdef reverse(self):
		"""
		Create a directed graph with identical content in which edge directions have been reversed.

		.. note::
		 	Node and edge indices are preserved in the reversed graph.
		"""
		cdef DiGraph G = DiGraph()
		cdef int i,j,eidx,eidx2
		cdef double weight

		for i in range(self.next_node_idx):
			if self.node_info[i].exists:
				nobj = self.node_object(i)
				ndata = self.node_data_(i)

				# this way of adding nodes preserves the node index
				G.add_node_x(i,G.edge_list_capacity,G.edge_list_capacity,nobj,ndata)

		for eidx in range(self.next_edge_idx):
			if self.edge_info[eidx].exists:
				i = self.edge_info[eidx].src
				j = self.edge_info[eidx].tgt
				edata = self.edge_data_(eidx)
				weight = self.weight_(eidx)

				eidx2 = G.add_edge_x(eidx,j,i,edata,weight)

		return G

	cpdef np.ndarray[np.int_t] add_nodes(self,int num_nodes,node_obj_fxn=None):
		"""
		Add a specified set of nodes to the graph.

		**Args**:

			* ``num_nodes`` (int): the number of nodes to add to the graph.
			* ``node_obj_fxn [=None]`` (callable): a callable object (typically a function) that accepts a node index (an integer)
				and returns the object that will be used as the object for that node in the graph.  If this is not specified, then
				no node objects will be assigned to nodes.

		**Returns**:
			``np.ndarray[int,ndims=1]``, ``I``. The node indices of the nodes added where ``I[j]`` is the index of the jth node added by this
			function call.

		**Raises**:
			:py:exc:`ZenException`: if a node could not be added.
		"""
		cdef int nn_count
		cdef np.ndarray[np.int_t,ndim=1] indexes = np.empty( num_nodes, np.int)
		cdef int node_idx

		self.num_changes += 1

		for nn_count in range(num_nodes):

			# recycle an unused node index if possible
			if self.first_free_node != -1:
				node_idx = self.first_free_node
			else:
				node_idx = self.next_node_idx

			indexes[nn_count] = node_idx

			# add a node object if specified
			nobj = None
			if node_obj_fxn is not None:
				nobj = node_obj_fxn(node_idx)

			self.add_node_x(node_idx,self.edge_list_capacity,self.edge_list_capacity,nobj,None)

		return indexes

	cpdef int add_node(DiGraph self,nobj=None,data=None) except -1:
		"""
		Add a node to this graph.

		**Args**:

			* ``nobj [=None]``: the (optional) object that will be a convenient identifier for this node.
			* ``data [=None]``: an optional data object to associated with this node.

		**Returns**:
			``int``. The index of the new node.

		**Raises**:
			:py:exc:`ZenException`: if the node could not be added.
		"""
		cdef int node_idx

		# recycle an unused node index if possible
		if self.first_free_node != -1:
			node_idx = self.first_free_node
		else:
			node_idx = self.next_node_idx

		self.add_node_x(node_idx,self.edge_list_capacity,self.edge_list_capacity,nobj,data)

		return node_idx

	cdef add_to_free_node_list(self,int nidx):
		self.node_info[nidx].indegree = self.first_free_node
		self.node_info[nidx].in_capacity = -1

		if self.first_free_node != -1:
			self.node_info[self.first_free_node].in_capacity = nidx

		self.first_free_node = nidx

	cdef remove_from_free_node_list(self,int nidx):
		cdef int prev_free_node = self.node_info[nidx].in_capacity
		cdef int next_free_node = self.node_info[nidx].indegree

		if prev_free_node == -1:
			self.first_free_node = next_free_node
		else:
			self.node_info[prev_free_node].indegree = next_free_node

		if next_free_node != -1:
			self.node_info[next_free_node].in_capacity = prev_free_node

	cpdef add_node_x(DiGraph self,int node_idx,int in_edge_list_capacity,int out_edge_list_capacity,nobj,data):
		"""
		Adds a node to the graph with a specific node index.

		This function permits very high-performance population of the graph data structure
		with nodes by allowing the calling function to specify the node index and edge
		capacity of the node being added.  In general, this should only be done when the node indices
		have been obtained from a previously stored graph data structure.

		.. DANGER::
			This function should be used with great care because by specifying a node index, the
			calling function is forcing Zen to access specific parts of the memory allocated for nodes.
			Unless you are writing high-performance network loading code, you should not be calling
			this function directly.

			When used incorrectly, this method call can irreparably damage the integrity of the graph object,
			leading to incorrect results or, more likely, segmentation faults.

		**Args**:

			* ``node_idx`` (int): the node index this node should be assigned.
			* ``in_edge_list_capacity`` (int): the number of entries that should be allocated in the edge list for this node.
			* ``out_edge_list_capacity`` (int): the number of entries that should be allocated in the edge list for this node.
			* ``nobj``: the node object that will be associated with this node.  If ``None``, then no object will be
				assigned to this node.
			* ``data``: the data object that will be associated with this node.  If ``None``, then no data will be
				assigned to this node.

		**Raises**:
			:py:exc:`ZenException`: if the node index is already in use or the node object is not unique.
		"""
		cdef int i

		if node_idx < self.node_capacity and self.node_info[node_idx].exists:
			raise ZenException, 'Adding node at index %d will overwrite an existing node' % node_idx

		if node_idx >= self.next_node_idx:
			# if we got in here, then we must be in the
			# domain beyond the next_node_idx - these are the nodes
			# that aren't in the free list.

			# add all nodes that are being skipped over to the free node list
			for i in range(self.next_node_idx,imin(node_idx,self.node_capacity)):
				self.add_to_free_node_list(i)

			self.next_node_idx = node_idx + 1
		else:
			# Fix the hole in the free node list that will be created
			self.remove_from_free_node_list(node_idx)

		# update the max_node_idx to keep track of the maximum value existing index
		if node_idx > self.max_node_idx:
			self.max_node_idx = node_idx

		self.num_changes += 1

		if nobj is not None:
			if nobj in self.node_idx_lookup:
				raise ZenException, 'Node object "%s" is already in use' % str(nobj)
				
			self.node_idx_lookup[nobj] = node_idx
			self.node_obj_lookup[node_idx] = nobj

		if data is not None:
			self.node_data_lookup[node_idx] = data

		# grow the node_info array as necessary
		cdef int new_node_capacity
		while node_idx >= self.node_capacity:
			new_node_capacity = <int> ceil( float(self.node_capacity) * self.node_grow_factor)
			if node_idx >= new_node_capacity:
				new_node_capacity = node_idx + 1

			self.node_info = <NodeInfo*> stdlib.realloc(self.node_info, sizeof_NodeInfo*new_node_capacity)
			for i in range(self.node_capacity,new_node_capacity):
				self.node_info[i].exists = False

			# add all the newly allocated empty nodes to the free node list
			for i in range(self.node_capacity,node_idx):
				self.add_to_free_node_list(i)

			self.node_capacity = new_node_capacity

		self.node_info[node_idx].exists = True
		self.node_info[node_idx].indegree = 0
		self.node_info[node_idx].outdegree = 0

		# initialize edge lists
		self.node_info[node_idx].inelist = <int*> stdlib.malloc(sizeof(int) * self.edge_list_capacity)
		self.node_info[node_idx].in_capacity = self.edge_list_capacity
		self.node_info[node_idx].outelist = <int*> stdlib.malloc(sizeof(int) * self.edge_list_capacity)
		self.node_info[node_idx].out_capacity = self.edge_list_capacity

		self.num_nodes += 1
		
		# notify listeners if necessary
		if self.num_graph_listeners > 0:
			for listener in self.graph_listeners:
				listener.node_added(node_idx,nobj,data)
				
		return

	def __contains__(DiGraph self,nobj):
		"""
		Return ``True`` if object ``nobj`` is associated with a node in this graph.
		"""
		return nobj in self.node_idx_lookup

	cpdef int node_idx(DiGraph self,nobj) except -1:
		"""
		Return the index of the node with node object ``nobj``.
		"""
		return self.node_idx_lookup[nobj]

	cpdef node_object(DiGraph self,int nidx):
		"""
		Return the object associated with node having index ``nidx``.  If no object is associated, then ``None`` is returned.
		"""
		if nidx in self.node_obj_lookup:
			return self.node_obj_lookup[nidx]
		else:
			return None

	cpdef set_node_object(self,curr_node_obj,new_node_obj):
		"""
		Change the node object associated with a specific node.

		.. note::
			The new object must be unique among all other node objects.

		**Args**:

			* ``curr_node_obj``: the current node object to change.
			* ``new_node_obj``: the object to replace the current node object with.

		**Raises**:
			:py:exc:`ZenException`: if the new key object is not unique in the graph.
		"""
		if curr_node_obj == new_node_obj:
			return
		else:
			if new_node_obj in self.node_idx_lookup:
				raise ZenException, 'Node object %s is not unique' % str(new_node_obj)
			else:
				self.node_idx_lookup[new_node_obj] = self.node_idx_lookup[curr_node_obj]
				del self.node_idx_lookup[curr_node_obj]
				self.node_obj_lookup[self.node_idx_lookup[new_node_obj]] = new_node_obj

	cpdef set_node_object_(self,node_idx,new_node_obj):
		"""
		Change the node object associated with a specific node.

		.. note::
			The new object must be unique among all other node objects.

		**Args**:

			* ``node_idx``: the index of the node to set the object for.
			* ``new_node_obj``: the object to replace the current node object with.

		**Raises**:
			:py:exc:`ZenException`: if the new key object is not unique in the graph.
		"""
		if node_idx >= self.node_capacity or not self.node_info[node_idx].exists:
			raise ZenException, 'Invalid node idx %d' % node_idx

		if new_node_obj == self.node_object(node_idx):
			return

		if new_node_obj in self.node_idx_lookup:
			raise ZenException, 'Node object %s is not unique' % str(new_node_obj)

		if node_idx in self.node_obj_lookup:
			del self.node_idx_lookup[self.node_obj_lookup[node_idx]]
		self.node_idx_lookup[new_node_obj] = node_idx
		self.node_obj_lookup[node_idx] = new_node_obj

	cpdef node_data(DiGraph self,nobj):
		"""
		Return the data object associated with node having object identifier ``nobj``.
		"""
		return self.node_data_(self.node_idx_lookup[nobj])

	cpdef set_node_data(DiGraph self,nobj,data):
		"""
		Associate a new data object with a specific node in the network.
		If data is None, then any data associated with the node is deleted.

		**Args**:

			* ``nobj``: the node object identifying the node whose data association is being changed.
			* ``data``: the data object to associate.  If ``None``, then any data object currently
				associated with this node will be deleted.
		"""
		self.set_node_data_(self.node_idx_lookup[nobj],data)

	cpdef set_node_data_(DiGraph self,int nidx,data):
		"""
		Associate a new data object with a specific node in the network.
		If data is None, then any data associated with the node is deleted.

		**Args**:

			* ``nidx``: the index of the node whose data association is being changed.
			* ``data``: the data object to associate.  If ``None``, then any data object currently
				associated with this node will be deleted.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node index %d' % nidx

		if data == None:
			if nidx in self.node_data_lookup:
				del self.node_data_lookup[nidx]
		else:
			self.node_data_lookup[nidx] = data

	cpdef node_data_(DiGraph self,int nidx):
		"""
		Return the data object associated with node having index ``nidx``.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node index %d' % nidx

		if nidx in self.node_data_lookup:
			return self.node_data_lookup[nidx]
		else:
			return None

	cpdef nodes_iter(DiGraph self,data=False):
		"""
		Return an iterator over all the nodes in the graph.

		By default, the iterator yields node objects.  If ``data`` is ``True``,
		then the iterator yields tuples ``(node_object,node_data)``.
		"""
		return NodeIterator(self,False,data,True)

	cpdef nodes_iter_(DiGraph self,obj=False,data=False):
		"""
		Return an iterator over all the nodes in the graph.

		By default, the iterator yields node indices.  If either ``obj`` or ``data`` are ``True``, then
		tuples are yielded.  For example::

			for nidx in G.node_iter_():
				print 'Node index:',nidx

			for nidx,obj in G.node_iter_(obj=True):
				print 'Node index:',nidx,'Node object:',obj

			for nidx,obj,data in G.node_iter_(True,True):
				print nidx,obj,data

		"""
		return NodeIterator(self,obj,data,False)

	cpdef nodes(DiGraph self,data=False):
		"""
		Return a list of the node objects in the graph.

		.. note::
			This method is only valid for graphs in which all nodes have a node object.

		By default, the list contains node objects for all nodes.  If ``data`` is ``True``,
		then the list contains tuples containing the node object and associated data.
		"""
		result = []

		cdef int idx = 0
		cdef int i = 0
		while idx < self.num_nodes:
			if self.node_info[i].exists:
				if i not in self.node_obj_lookup:
					raise ZenException, 'Node (idx=%d) is missing object' % i
				if data:
					if i in self.node_data_lookup:
						result.append( (self.node_obj_lookup[i],self.node_data_lookup[i]) )
					else:
						result.append( (self.node_obj_lookup[i],None) )
				else:
					result.append(self.node_obj_lookup[i])
				idx += 1

			i += 1

		return result

	cpdef nodes_(DiGraph self,obj=False,data=False):
		"""
		Return a numpy array of the nodes.

		By default, the array is 1-D and contains only node indices.  If either ``obj`` or ``data``
		are ``True``, then the result is a 2-D matrix in which the additional columns contain
		the node object and/or data.

		.. note::
			If ``obj`` and ``data`` are both ``False``, then the numpy array returned has type ``int``.  When used
			with cython, this fact can be used to dramatically increase the speed of code iterating
			over a graph's nodes.
		"""
		ndim = 1 if not data and not obj else (2 if not data or not obj else 3)

		result = None
		if ndim > 1:
			result = numpy.empty( (self.num_nodes,ndim), dtype=numpy.object_)
		else:
			result = numpy.zeros( self.num_nodes, dtype=numpy.int_)

		cdef int idx = 0
		cdef int i = 0

		if ndim == 1:
			for i in range(self.next_node_idx):
				if self.node_info[i].exists:
					result[idx] = i
					idx += 1
		elif obj and not data:
			for i in range(self.next_node_idx):
				if self.node_info[i].exists:
					result[idx,0] = i
					if i in self.node_obj_lookup:
						result[idx,1] = self.node_obj_lookup[i]
					else:
						result[idx,1] = None
					idx += 1

				i += 1
		elif data and not obj:
			for i in range(self.next_node_idx):
				if self.node_info[i].exists:
					result[idx,0] = i
					if i in self.node_data_lookup:
						result[idx,1] = self.node_data_lookup[i]
					else:
						result[idx,1] = None
					idx += 1

				i += 1
		elif data and not obj:
			for i in range(self.next_node_idx):
				if self.node_info[i].exists:
					result[idx,0] = i

					if i in self.node_obj_lookup:
						result[idx,1] = self.node_obj_lookup[i]
					else:
						result[idx,1] = None

					if i in self.node_data_lookup:
						result[idx,2] = self.node_data_lookup[i]
					else:
						result[idx,2] = None

					idx += 1

				i += 1

		return result

	cpdef rm_node(DiGraph self,nobj):
		"""
		Remove the node associated with node object ``nobj``.  Any edges incident to the node are also removed
		from the graph.
		"""
		self.rm_node_(self.node_idx_lookup[nobj])

	cpdef rm_node_(DiGraph self,int nidx):
		"""
		Remove the node with index ``nidx``. Any edges incident to the node are also removed from the graph.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node index %d' % nidx

		cdef int i

		self.num_changes += 1

		# disconnect all inbound edges
		for i in range(self.node_info[nidx].indegree-1,-1,-1):
			eidx = self.node_info[nidx].inelist[i]

			# remove the edge
			self.rm_edge_(eidx)

		# disconnect all outbound edges
		for i in range(self.node_info[nidx].outdegree-1,-1,-1):
			eidx = self.node_info[nidx].outelist[i]

			# remove the edge
			self.rm_edge_(eidx)

		# free up all the data structures
		self.node_info[nidx].exists = False
		stdlib.free(self.node_info[nidx].inelist)
		stdlib.free(self.node_info[nidx].outelist)

		if nidx in self.node_data_lookup:
			del self.node_data_lookup[nidx]

		if nidx in self.node_obj_lookup:
			nobj = self.node_obj_lookup[nidx]
			del self.node_obj_lookup[nidx]
			del self.node_idx_lookup[nobj]

		# keep track of the free node
		self.add_to_free_node_list(nidx)

		# update the max_node_idx value (if necessary)
		if nidx == self.max_node_idx:
			while self.max_node_idx >= 0 and not self.node_info[self.max_node_idx].exists:
				self.max_node_idx -= 1

		self.num_nodes -= 1
		
		# notify listeners if necessary
		if self.num_graph_listeners > 0:
			for listener in self.graph_listeners:
				listener.node_removed(nidx,nobj)

	cpdef degree(DiGraph self,nobj):
		"""
		Return the degree of node with object ``nobj``.
		"""
		return self.degree_(self.node_idx_lookup[nobj])

	cpdef degree_(DiGraph self,int nidx):
		"""
		Return the degree of node with index ``nidx``.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node index %d' % nidx

		return self.node_info[nidx].indegree + self.node_info[nidx].outdegree

	cpdef in_degree(DiGraph self,nobj):
		"""
		Return the in-degree of node with object ``nobj``.
		"""
		return self.in_degree_(self.node_idx_lookup[nobj])

	cpdef in_degree_(DiGraph self,int nidx):
		"""
		Return the in-degree of node with index ``nidx``.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node index %d' % nidx

		return self.node_info[nidx].indegree

	cpdef out_degree(DiGraph self,nobj):
		"""
		Return the out-degree of node with object ``nobj``.
		"""
		return self.out_degree_(self.node_idx_lookup[nobj])

	cpdef out_degree_(DiGraph self,int nidx):
		"""
		Return the out-degree of node with index ``nidx``.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node index %d' % nidx

		return self.node_info[nidx].outdegree

	def __getitem__(self,nobj):
		"""
		Get the data for the node associated with ``nobj``.
		"""
		return self.node_data_lookup[self.node_idx_lookup[nobj]]

	def __len__(self):
		"""
		Return the number of nodes in the graph.
		"""
		return self.num_nodes

	def size(self):
		"""
		Return the number of edges in the graph.
		"""
		return self.num_edges

	cpdef int add_edge(self, src, tgt, data=None, double weight=1) except -1:
		"""
		Add an edge to the graph.

		As a convenience, if ``src`` or ``tgt`` are not valid node objects, they will be added to the graph
		and then the edge will be added.::

			G = DiGraph()

			print len(G) # prints '0'
			G.add_edge(1,2) # First nodes 1 and 2 will be added, then the edge will be added
			print len(G) # prints '2' since there are now two nodes in the graph.


		**Args**:

			* ``src``: the node from which the edge originates
			* ``tgt``: the node at which the edge terminates
			* ``data [=None]``: an optional data object to associate with the edge
			* ``weight [=1]`` (float): the weight of the edge.

		**Returns**:
			``integer``. The index for the newly created edge.

		**Raises**:
			:py:exc:`ZenException`: if the edge already exists in the graph.

		"""
		cdef int nidx1, nidx2

		if src not in self.node_idx_lookup:
			nidx1 = self.add_node(src)
		else:
			nidx1 = self.node_idx_lookup[src]

		if tgt not in self.node_idx_lookup:
			nidx2 = self.add_node(tgt)
		else:
			nidx2 = self.node_idx_lookup[tgt]

		result = self.add_edge_(nidx1,nidx2,data,weight)

		return result

	cpdef int add_edge_(DiGraph self, int src, int tgt, data=None, double weight=1) except -1:
		"""
		Add an edge to the graph.

		This version of the edge addition functionality uses node indices (not node objects).
		Unlike in :py:method:``.add_edge``, if ``src`` or ``tgt`` are not valid node indices, then an
		exception will be raised.

		**Args**:

			* ``src`` (int): the node from which the edge originates. This should be a node index.
			* ``tgt`` (int): the node at which the edge terminates. This should be a node index.
			* ``data [=None]``: an optional data object to associate with the edge
			* ``weight [=1]`` (float): the weight of the edge.

		**Returns**:
			``integer``. The index for the newly created edge.

		**Raises**:
			:py:exc:`ZenException`: if the edge already exists in the graph or if either of the node indices are invalid.
		"""
		cdef int i
		self.num_changes += 1

		if src >= self.node_capacity or not self.node_info[src].exists or tgt >= self.node_capacity or not self.node_info[tgt].exists:
			raise ZenException, 'Both source and destination nodes must exist (%d,%d)' % (src,tgt)

		cdef int eidx

		# recycle an edge index if possible
		if self.first_free_edge != -1:
			eidx = self.first_free_edge
		else:
			eidx = self.next_edge_idx

		self.add_edge_x(eidx,src,tgt,data,weight)

		return eidx

	cdef add_to_free_edge_list(self,int eidx):
		self.edge_info[eidx].src = self.first_free_edge
		self.edge_info[eidx].tgt = -1

		if self.first_free_edge != -1:
			self.edge_info[self.first_free_edge].tgt = eidx

		self.first_free_edge = eidx

	cdef remove_from_free_edge_list(self,int eidx):
		cdef int prev_free_edge = self.edge_info[eidx].tgt
		cdef int next_free_edge = self.edge_info[eidx].src

		if prev_free_edge == -1:
			self.first_free_edge = next_free_edge
		else:
			self.edge_info[prev_free_edge].src = next_free_edge

		if next_free_edge != -1:
			self.edge_info[next_free_edge].tgt = prev_free_edge

	cpdef int add_edge_x(DiGraph self, int eidx, int src, int tgt, data, double weight) except -1:
		"""
		Adds an edge to the graph with a specific edge index.

		This function permits very high-performance population of the graph data structure
		with edges by allowing the calling function to specify the edge index of the edge being added.
		In general, this should only be done when the edge indices have been obtained from a previously
		stored graph data structure.

		.. DANGER::
			This function should be used with great care because by specifying a edge index, the
			calling function is forcing Zen to access specific parts of the memory allocated for edge.
			Unless you are writing high-performance network loading code, you should not be calling
			this function directly.

			When used incorrectly, this method call can irreparably damage the integrity of the graph object,
			leading to incorrect results or, more likely, segmentation faults.

		**Args**:

			* ``eidx`` (int): the edge index this node should be assigned.
			* ``src`` (int): the node from which the edge originates. This should be a node index.
			* ``tgt`` (int): the node at which the edge terminates. This should be a node index.
			* ``nobj``: the node object that will be associated with this node.  If ``None``, then no object will be
				assigned to this node.
			* ``data``: the data object that will be associated with this node.  If ``None``, then no data will be
				assigned to this node.

		**Raises**:
			:py:exc:`ZenException`: if the edge already exists in the graph, the edge index is already in use, or either of the
			node indices are invalid.
		"""
		if eidx < self.edge_capacity and self.edge_info[eidx].exists:
			raise ZenException, 'Adding edge at index %d will overwrite an existing edge' % eidx

		# grow the info array
		cdef int new_edge_capacity
		while eidx >= self.edge_capacity:
			new_edge_capacity = <int> ceil( <float>self.edge_capacity * self.edge_grow_factor)
			self.edge_info = <EdgeInfo*> stdlib.realloc( self.edge_info, sizeof_EdgeInfo * new_edge_capacity)
			for i in range(self.edge_capacity,new_edge_capacity):
				self.edge_info[i].exists = False

			self.edge_capacity = new_edge_capacity

		####
		# connect up the edges to nodes

		# Note: this is where duplicate edges are detected.  So
		# an exception can be thrown by these two functions.  Hence,
		# it's necessary to make modifications to the edge list *AFTER*
		# these functions both successfully return.

		# source
		self.__insert_edge_into_outelist(src,eidx,tgt)

		# destination
		self.__insert_edge_into_inelist(tgt,eidx,src)

		# obtain the node index from the
		if eidx >= self.next_edge_idx:
			# if we got in here, then we must be in the
			# domain beyond the next_edge_idx - these are the edges
			# that aren't in the free list.

			# add all edges that are being skipped over to the free edge list
			for i in range(self.next_edge_idx,eidx):
				self.add_to_free_edge_list(i)

			self.next_edge_idx = eidx + 1
		else:
			# Fix the hole in the free node list that will be created
			self.remove_from_free_edge_list(eidx)

		# update the max_edge_idx to keep track of the maximum value existing index
		if eidx > self.max_edge_idx:
			self.max_edge_idx = eidx

		# set the data if provided
		if data is not None:
			self.edge_data_lookup[eidx] = data

		### Add edge info
		self.edge_info[eidx].exists = True
		self.edge_info[eidx].src = src
		self.edge_info[eidx].tgt = tgt
		self.edge_info[eidx].weight = weight

		#####
		# Done
		self.num_edges += 1
		
		# notify listeners if necessary
		if self.num_graph_listeners > 0:
			for listener in self.graph_listeners:
				listener.edge_added(eidx,src,tgt,data,weight)

		return eidx

	cdef __insert_edge_into_outelist(DiGraph self, int src, int eidx, int tgt):
		cdef int new_capacity
		cdef int num_edges = self.node_info[src].outdegree
		cdef int pos = 0
		cdef double elist_len = <double> self.node_info[src].out_capacity
		if num_edges >= elist_len:
			new_capacity = <int>ceil(elist_len * self.edge_list_grow_factor)
			self.node_info[src].outelist = <int*> stdlib.realloc(self.node_info[src].outelist, sizeof(int) * new_capacity)
			self.node_info[src].out_capacity = new_capacity
		cdef int* elist = self.node_info[src].outelist
		pos = self.find_outelist_insert_pos(elist,num_edges,tgt)
		if pos == num_edges:
			elist[pos] = eidx
		elif self.edge_info[elist[pos]].tgt == tgt:
			raise ZenException, 'Duplicate edges (%d,%d) are not permitted in a DiGraph' % (src,tgt)
		else:
			memmove(elist + (pos+1),elist + pos,(num_edges-pos) * sizeof(int))
			elist[pos] = eidx
		self.node_info[src].outdegree += 1

	cdef __insert_edge_into_inelist(DiGraph self, int tgt, int eidx, int src):
		cdef int new_capacity
		cdef int pos = 0
		cdef int num_edges = self.node_info[tgt].indegree
		cdef double elist_len = <double> self.node_info[tgt].in_capacity
		if num_edges >= elist_len:
			new_capacity = <int>ceil(elist_len * self.edge_list_grow_factor)
			self.node_info[tgt].inelist = <int*> stdlib.realloc(self.node_info[tgt].inelist, sizeof(int) * new_capacity)
			self.node_info[tgt].in_capacity = new_capacity
		cdef int* elist = self.node_info[tgt].inelist
		pos = self.find_inelist_insert_pos(elist,num_edges,src)
		if pos == num_edges:
			elist[pos] = eidx
		else:
			memmove(elist + (pos+1),elist + pos,(num_edges-pos) * sizeof(int))
			elist[pos] = eidx
		self.node_info[tgt].indegree += 1

	cdef int find_inelist_insert_pos(DiGraph self, int* elist, int elist_len, int nidx):
		"""
		Perform a binary search for the insert position
		"""
		cdef int pos = <int> floor(elist_len/2)

		if elist_len == 0:
			return 0

		if pos == 0:
			if self.edge_info[elist[pos]].src < nidx:
				return elist_len
			else:
				return 0

		while True:
			if pos == 0:
				return 0
			elif pos == (elist_len-1) and self.edge_info[elist[pos]].src < nidx:
				return elist_len
			elif (self.edge_info[elist[pos-1]].src < nidx and self.edge_info[elist[pos]].src >= nidx):
				return pos
			else:
				if self.edge_info[elist[pos]].src < nidx:
					pos = pos + <int> floor((elist_len-pos)/2)
				else:
					elist_len = pos
					pos = <int> floor(pos/2)

	cdef int find_outelist_insert_pos(DiGraph self, int* elist, int elist_len, int nidx):
		"""
		Perform a binary search for the insert position
		"""
		cdef int pos = <int> floor(elist_len/2)

		if elist_len == 0:
			return 0

		if pos == 0:
			if self.edge_info[elist[pos]].tgt < nidx:
				return elist_len
			else:
				return 0

		while True:
			if pos == 0:
				return 0
			elif pos == (elist_len-1) and self.edge_info[elist[pos]].tgt < nidx:
				return elist_len
			elif (self.edge_info[elist[pos-1]].tgt < nidx and self.edge_info[elist[pos]].tgt >= nidx):
				return pos
			else:
				if self.edge_info[elist[pos]].tgt < nidx:
					pos = pos + <int> floor((elist_len-pos)/2)
				else:
					elist_len = pos
					pos = <int> floor(pos/2)

	cpdef rm_edge(DiGraph self,src,tgt):
		"""
		Remove the edge connecting node object ``src`` to ``tgt``.

		**Raises**:

			* :py:exc:`ZenException`: if the edge index is invalid.
			* :py:exc:`KeyError`: if the node objects are invalid.
		"""
		self.rm_edge_(self.edge_idx(src,tgt))

	cpdef rm_edge_(DiGraph self,int eidx):
		"""
		Remove the edge with index ``eidx``.

		**Raises**:
			:py:exc:`ZenException`: if ``eid`` is an invalid edge index.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge index %d' % eidx

		cdef int i

		self.num_changes += 1

		#####
		# remove entries in source and target
		cdef int src = self.edge_info[eidx].src
		cdef int tgt = self.edge_info[eidx].tgt

		# in src
		self.__remove_edge_from_outelist(src,eidx,tgt)

		# in tgt
		self.__remove_edge_from_inelist(tgt,eidx,src)

		# remove actual data structure
		self.edge_info[eidx].exists = False

		if eidx in self.edge_data_lookup:
			data = self.edge_data_lookup[eidx]
			del self.edge_data_lookup[eidx]

		# add the edge to the list of free edges
		self.add_to_free_edge_list(eidx)

		# update the max_edge_idx value (if necessary)
		if eidx == self.max_edge_idx:
			while self.max_edge_idx >= 0 and not self.edge_info[self.max_edge_idx].exists:
				self.max_edge_idx -= 1

		self.num_edges -= 1

		# notify listeners if necessary
		if self.num_graph_listeners > 0:
			for listener in self.graph_listeners:
				listener.edge_removed(eidx,src,tgt)
				
		return

	cdef __remove_edge_from_outelist(DiGraph self, int src, int eidx, int tgt):
		cdef int i = self.find_outelist_insert_pos(self.node_info[src].outelist,self.node_info[src].outdegree,tgt)
		memmove(&self.node_info[src].outelist[i],&self.node_info[src].outelist[i+1],(self.node_info[src].outdegree-i-1)*sizeof(int))
		self.node_info[src].outdegree -= 1

	cdef __remove_edge_from_inelist(DiGraph self, int tgt, int eidx, int src):
		cdef int i = self.find_inelist_insert_pos(self.node_info[tgt].inelist,self.node_info[tgt].indegree,src)
		memmove(&self.node_info[tgt].inelist[i],&self.node_info[tgt].inelist[i+1],(self.node_info[tgt].indegree-i-1)*sizeof(int))
		self.node_info[tgt].indegree -= 1

	cpdef endpoints(DiGraph self,int eidx):
		"""
		Return the node objects at the endpoints of the edge with index ``eidx``.
		"""
		if not self.edge_info[eidx].exists:
			raise ZenException, 'Edge with ID %d does not exist' % eidx

		return self.node_obj_lookup[self.edge_info[eidx].src], self.node_obj_lookup[self.edge_info[eidx].tgt]

	cpdef endpoints_(DiGraph self,int eidx):
		"""
		Return the node indices at the endpoints of the edge with index ``eidx``.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge index %d' % eidx

		return self.edge_info[eidx].src, self.edge_info[eidx].tgt

	cpdef endpoint(DiGraph self,int eidx,u):
		"""
		Return the object for the node (not u) that is the endpoint of this edge.

		.. note::
			For performance reasons, no check is done to ensure that u is an endpoint of the edge.

		**Args**:

			* ``eidx`` (int): a valid edge index.
			* ``u``: the object for one endpoint of the edge with index ``eidx``.

		**Returns**:
			``object``. The object for the node that is the other endpoint of edge ``eidx``.
		"""
		if u not in self.node_idx_lookup:
			raise ZenException, 'Invalid node object %s' % str(u)
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx

		return self.node_object(self.endpoint_(eidx,self.node_idx_lookup[u]))

	cpdef int endpoint_(DiGraph self,int eidx,int u) except -1:
		"""
		Return the index for the node (not u) that is the endpoint of this edge.

		.. note::
			For performance reasons, no check is done to ensure that u is an endpoint of the edge.

		**Args**:

			* ``eidx`` (int): a valid edge index.
			* ``u`` (int): the index for one endpoint of the edge with index ``eidx``.

		**Returns**:
			``integer``. The index for the node that is the other endpoint of edge ``eidx``.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx

		if u == self.edge_info[eidx].src:
			return self.edge_info[eidx].tgt
		else:
			return self.edge_info[eidx].src

	cpdef src(DiGraph self,int eidx):
		"""
		Return the object for the node from which edge ``eidx`` originates.
		"""
		return self.node_object(self.src_(eidx))

	cpdef int src_(DiGraph self,int eidx) except -1:
		"""
		Return the index for the node from which edge ``eidx`` originates.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx

		return self.edge_info[eidx].src

	cpdef tgt(DiGraph self,int eidx):
		"""
		Return the object for the node at which edge ``eidx`` terminates.
		"""
		return self.node_object(self.tgt_(eidx))

	cpdef int tgt_(DiGraph self,int eidx) except -1:
		"""
		Return the index for the node at which edge ``eidx`` terminates.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx

		return self.edge_info[eidx].tgt

	cpdef set_weight(DiGraph self,src,tgt,double w):
		"""
		Set the weight of the edge connecting node ``src`` to ``tgt`` (node objects) to ``w``.
		"""
		self.set_weight_(self.edge_idx(src,tgt),w)

	cpdef set_weight_(DiGraph self,int eidx,double w):
		"""
		Set the weight of the edge with index ``eidx`` to ``w``.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge index %d' % eidx

		self.edge_info[eidx].weight = w

	cpdef double weight(DiGraph self,src,tgt):
		"""
		Return the weight of the edge connecting node ``src`` to ``tgt`` (node objects).
		"""
		return self.weight_(self.edge_idx(src,tgt))

	cpdef double weight_(DiGraph self,int eidx):
		"""
		Return the weight of the edge with index ``eidx``.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge index %d' % eidx

		return self.edge_info[eidx].weight

	cpdef set_edge_data(DiGraph self,src,tgt,data):
		"""
		Associate a data object with the edge between nodes ``src`` and ``tgt`` (node objects).

		The value of ``data`` will replace any data object currently associated with the edge.
		If data is None, then any data associated with the edge is deleted.
		"""
		self.set_edge_data_(self.edge_idx(src,tgt),data)

	cpdef edge_data(DiGraph self,src,tgt):
		"""
		Return the data associated with the edge between ``src`` and ``tgt`` (node objects).
		"""
		return self.edge_data_(self.edge_idx(src,tgt))

	cpdef set_edge_data_(DiGraph self,int eidx,data):
		"""
		Associate a data object with the edge with index ``eidx``.

		The value of ``data`` will replace any data object currently associated with the edge.
		If data is None, then any data associated with the edge is deleted.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge index %d' % eidx

		if data == None:
			if eidx in self.edge_data_lookup:
				del self.edge_data_lookup[eidx]
		else:
			self.edge_data_lookup[eidx] = data

	cpdef edge_data_(DiGraph self,int eidx):
		"""
		Return the data associated with the edge with index ``eidx``.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge index %d' % eidx

		if eidx in self.edge_data_lookup:
			return self.edge_data_lookup[eidx]
		else:
			return None

	cpdef bint has_edge(DiGraph self,src,tgt):
		"""
		Return ``True`` if the graph contains an edge between ``src`` and ``tgt`` (node objects).
		If either node object is not in the graph, this method returns ``False``.
		"""
		if src not in self.node_idx_lookup:
			return False

		if tgt not in self.node_idx_lookup:
			return False

		src = self.node_idx_lookup[src]
		tgt = self.node_idx_lookup[tgt]

		return self.has_edge_(src,tgt)

	cpdef bint has_edge_(DiGraph self,int src,int tgt):
		"""
		Return ``True`` if the graph contains an edge between ``src`` and ``tgt`` (node indices).

		**Raises**:
			:py:exc:`ZenException`: if either ``src`` or ``tgt`` are invalid node indices.
		"""
		if src >= self.node_capacity or not self.node_info[src].exists:
			raise ZenException, 'Invalid source index %d' % src

		if tgt >= self.node_capacity or not self.node_info[tgt].exists:
			raise ZenException, 'Invalid target index %d' % tgt

		cdef int num_edges = self.node_info[src].outdegree
		cdef int* elist = self.node_info[src].outelist
		cdef int pos = self.find_outelist_insert_pos(elist,self.node_info[src].outdegree,tgt)
		return pos < self.node_info[src].outdegree and self.edge_info[elist[pos]].tgt == tgt

	cpdef edge_idx(DiGraph self, src, tgt, data=False):
		"""
		Return the edge index for the edge between ``src`` and ``tgt`` (node objects).
		"""
		src = self.node_idx_lookup[src]
		tgt = self.node_idx_lookup[tgt]
		return self.edge_idx_(src,tgt,data)

	cpdef edge_idx_(DiGraph self, int src, int tgt, data=False):
		"""
		Return the edge index for the edge between ``src`` and ``tgt`` (node indices).
		"""
		if src >= self.node_capacity or not self.node_info[src].exists:
			raise ZenException, 'Invalid source index %d' % src

		if tgt >= self.node_capacity or not self.node_info[tgt].exists:
			raise ZenException, 'Invalid target index %d' % tgt

		cdef int num_edges = self.node_info[src].outdegree
		cdef int* elist = self.node_info[src].outelist
		cdef int pos = self.find_outelist_insert_pos(elist,self.node_info[src].outdegree,tgt)

		if pos < self.node_info[src].outdegree and self.edge_info[elist[pos]].tgt == tgt:
			if data is True:
				return elist[pos], self.edge_data_lookup[elist[pos]]
			else:
				return elist[pos]
		else:
			raise ZenException, 'Edge (%d,%d) does not exist.' % (src,tgt)

	cpdef edges_iter(DiGraph self,nobj=None,data=False,weight=False):
		"""
		Return an iterator over edges in the graph.

		By default, the iterator will cover all edges in the graph, returning each
		edge as the tuple ``(u,v)``, where ``u`` and ``v`` are (node object) endpoints of the edge.

		**Args**:

			* ``nobj [=None]``: if ``nobj`` is specified (not ``None``), then the edges touching the node with
				object ``nobj`` are iterated over.

			* ``data [=False]`` (boolean): if ``True``, then the iterator adds object associated with the edge
			 	into the tuple returned (e.g., ``(u,v,d)``).

			* ``weight [=False]`` (boolean): 	if ``True``, then the iterator adds the weight of the edge
				into the tuple returned (e.g., ``(u,v,w)`` or ``(u,v,data,w)`` depending on the value of the ``data`` argument).

		Consider the following code which shows some of the different usages::

			G = DiGraph()
			G.add_edge(1,2,data='e1')
			G.add_edge(2,3,data='e2')
			G.add_edge(3,1,data='e3')

			print len(list(G.edges_iter())) # this prints 3 - there are 3 edges in the graph
			print len(list(G.edges_iter(1))) # this prints 2 - there are 2 edges attached to node 1

			# this will print the endpoints and data for all edges in the graph
			for u,v,data in G.edges_iter(data=True):
				print u,v,data

			# this will print the endpoints and data for all edges in the graph
			for u,v,w in G.edges_iter(weight=True):
				print u,v,w
		"""
		if nobj is None:
			return AllEdgeIterator(self,weight,data,True)
		else:
			return NodeEdgeIterator(self,self.node_idx_lookup[nobj],ITER_BOTH,weight,data,True)

	cpdef edges_iter_(DiGraph self,int nidx=-1,data=False,weight=False):
		"""
		Return an iterator over edges in the graph.

		By default, the iterator will cover all edges in the graph, returning each
		edge as the edge index.

		**Args**:

			* ``nidx [=-1]`` (int): if ``nidx`` is specified (``>= 0``), then the edges touching the node with
				index ``nidx`` are iterated over.

			* ``data [=False]`` (boolean): if ``True``, then the iterator returns a tuple containing the edge index
				and the data associated with the edge (e.g., ``(eidx,d)``).

			* ``weight [=False]`` (boolean): 	if ``True``, then the iterator 	returns a tuple containing the edge index
				and the weight of the edge (e.g., ``(eidx,w)`` or ``(eidx,d,w)`` depending on the value of the ``data`` argument).

		Consider the following code which shows some of the different usages::

			G = DiGraph()
			G.add_edge(1,2,data='e1')
			G.add_edge(2,3,data='e2')
			G.add_edge(3,1,data='e3')

			print len(list(G.edges_iter_())) # this prints 3 - there are 3 edges in the graph
			print len(list(G.edges_iter_(G.node_idx(1)))) # this prints 2 - there are 2 edges attached to node 1

			# this will print the endpoints and data for all edges in the graph
			for eidx,data in G.edges_iter_(data=True):
				print eidx,data

			# this will print the endpoints and data for all edges in the graph
			for eidx,w in G.edges_iter_(weight=True):
				print eidx,w
		"""
		if nidx == -1:
			return AllEdgeIterator(self,weight,data,False)
		else:
			if nidx >= self.node_capacity or not self.node_info[nidx].exists:
				raise ZenException, 'Invalid node idx %d' % nidx

			return NodeEdgeIterator(self,nidx,ITER_BOTH,weight,data,False)

	cpdef edges(DiGraph self,nobj=None,data=False,weight=False):
		"""
		Return a list of edges in the graph.

		By default, the list will contain all edges in the graph, each
		edge as the tuple ``(u,v)``, where ``u`` and ``v`` are (node object) endpoints of the edge.

		**Args**:

			* ``nobj [=None]``: if ``nobj`` is specified (not ``None``), then only the edges touching the node with
				object ``nobj`` are included in the list.

			* ``data [=False]`` (boolean): if ``True``, then the data object associated with the edge
			 	is added into the tuple returned for each edge (e.g., ``(u,v,d)``).

			* ``weight [=False]`` (boolean): 	if ``True``, then the weight of the edge is added
				into the tuple returned for each edge (e.g., ``(u,v,w)`` or ``(u,v,data,w)`` depending on the
				value of the ``data`` argument).

		Consider the following code which shows some of the different usages::

			G = DiGraph()
			G.add_edge(1,2,data='e1')
			G.add_edge(2,3,data='e2')
			G.add_edge(3,1,data='e3')

			print len(G.edges()) # this prints 3 - there are 3 edges in the graph
			print len(G.edges(1)) # this prints 2 - there are 2 edges attached to node 1

			# this will print the endpoints and data for all edges in the graph
			for u,v,data in G.edges(data=True):
				print u,v,data

			# this will print the endpoints and data for all edges in the graph
			for u,v,w in G.edges(weight=True):
				print u,v,w
		"""
		cdef int num_edges
		cdef int* elist
		cdef int i
		cdef nidx = -1
		cdef nidx2 = -1

		if nobj is not None:
			nidx = self.node_idx_lookup[nobj]

		# iterate over all edges
		result = []
		if nidx == -1:
			idx = 0
			for i in range(self.next_edge_idx):
				if self.edge_info[i].exists:
					if self.edge_info[i].src not in self.node_obj_lookup or self.edge_info[i].tgt not in self.node_obj_lookup:
						raise ZenException, 'Edge (idx=%d) does not have endpoints with node objects' % i

					if data is True:
						if weight is True:
							result.append( (self.node_obj_lookup[self.edge_info[i].src],self.node_obj_lookup[self.edge_info[i].tgt],self.edge_data_(i),self.edge_info[i].weight) )
						else:
							result.append( (self.node_obj_lookup[self.edge_info[i].src],self.node_obj_lookup[self.edge_info[i].tgt],self.edge_data_(i)) )
					else:
						if weight is True:
							result.append( (self.node_obj_lookup[self.edge_info[i].src],self.node_obj_lookup[self.edge_info[i].tgt],self.edge_info[i].weight) )
						else:
							result.append( (self.node_obj_lookup[self.edge_info[i].src],self.node_obj_lookup[self.edge_info[i].tgt]) )
					idx += 1

			return result
		else:
			num_edges = self.node_info[nidx].indegree
			elist = self.node_info[nidx].inelist
			for i in range(num_edges):
				idx = elist[i]
				if self.edge_info[idx].src not in self.node_obj_lookup or self.edge_info[idx].tgt not in self.node_obj_lookup:
					raise ZenException, 'Edge (idx=%d) does not have endpoints with node objects' % idx

				if data is True:
					if weight is True:
						result.append( (self.node_obj_lookup[self.edge_info[idx].src],self.node_obj_lookup[self.edge_info[idx].tgt],self.edge_data_(idx),self.edge_info[idx].weight) )
					else:
						result.append( (self.node_obj_lookup[self.edge_info[idx].src],self.node_obj_lookup[self.edge_info[idx].tgt],self.edge_data_(idx)) )
				else:
					if weight is True:
						result.append( (self.node_obj_lookup[self.edge_info[idx].src],self.node_obj_lookup[self.edge_info[idx].tgt],self.edge_info[idx].weight) )
					else:
						result.append( (self.node_obj_lookup[self.edge_info[idx].src],self.node_obj_lookup[self.edge_info[idx].tgt]) )

			# loop over out edges - this is copied verbatim from out_edges_iter(...)
			num_edges = self.node_info[nidx].outdegree
			elist = self.node_info[nidx].outelist
			for i in range(num_edges):
				idx = elist[i]
				if self.edge_info[idx].src not in self.node_obj_lookup or self.edge_info[idx].tgt not in self.node_obj_lookup:
					raise ZenException, 'Edge (idx=%d) does not have endpoints with node objects' % idx

				if data is True:
					if weight is True:
						result.append( (self.node_obj_lookup[self.edge_info[idx].src],self.node_obj_lookup[self.edge_info[idx].tgt],self.edge_data_(idx),self.edge_info[idx].weight) )
					else:
						result.append( (self.node_obj_lookup[self.edge_info[idx].src],self.node_obj_lookup[self.edge_info[idx].tgt],self.edge_data_(idx)) )
				else:
					if weight is True:
						result.append( (self.node_obj_lookup[self.edge_info[idx].src],self.node_obj_lookup[self.edge_info[idx].tgt],self.edge_info[idx].weight) )
					else:
						result.append( (self.node_obj_lookup[self.edge_info[idx].src],self.node_obj_lookup[self.edge_info[idx].tgt]) )

			return result

	cpdef edges_(DiGraph self,int nidx=-1,data=False,weight=False):
		"""
		Return a ``numpy.ndarray`` containing edges in the graph.

		By default, the return value is a 1D array, ``R``, that contains all edges in the graph, where ``R[i]`` is
		an edge index.

		**Args**:

			* ``nidx [=-1]`` (int): if ``nidx`` is specified (``>= 0``), then only the edges touching the node with
				index ``nidx`` are included in the array returned.

			* ``data [=False]`` (boolean): if ``True``, then the array will no longer be a 1D array.  A separate column will be added
			 	such that ``R[i,0]`` is the edge index and ``R[i,1]`` is the data object associated with the edge.

			* ``weight [=False]`` (boolean): 	if ``True``, then the array will no longer be a 1D array.  A separate column will be added
				such that ``R[i,0]`` is the edge index and ``R[i,1]`` is the weight of the edge.

		When additional columns are added, they will always be in the order edge index, data, weight.
		Consider the following code which shows some of the different usages::

			G = DiGraph()
			G.add_edge(1,2,data='e1')
			G.add_edge(2,3,data='e2')
			G.add_edge(3,1,data='e3')

			print G.edges_().shape # this prints (3,) - there are 3 edges in the graph and 1 column
			print G.edges_(G.node_idx(1)).shape # this prints (2,) - there are 2 edges attached to node 1

			# this will print the endpoints and data for all edges in the graph
			print G.edges_(data=True).shape # this prints (3,2)
			print G.edges_(weight=True).shape # this prints (3,2)
			print G.edges_(data=True,weight=True).shape # this prints (3,3)
		"""
		if nidx != -1 and (nidx >= self.node_capacity or not self.node_info[nidx].exists):
			raise ZenException, 'Invalid node idx %d' % nidx

		cdef int num_edges
		cdef int* elist
		cdef int i

		# iterate over all edges
		result = None
		if nidx == -1:
			if data and weight:
				result = numpy.empty( (self.num_edges, 3), dtype=numpy.object_)
			elif data or weight:
				result = numpy.empty( (self.num_edges, 2), dtype=numpy.object_)
			else:
				result = numpy.empty(self.num_edges, dtype=numpy.object_)

			idx = 0
			for i in range(self.next_edge_idx):
				if self.edge_info[i].exists:
					if data is True:
						if weight is True:
							result[idx,0] = i
							result[idx,1] = self.edge_data_(i)
							result[idx,2] = self.edge_info[i].weight
						else:
							result[idx,0] = i
							result[idx,1] = self.edge_data_(i)
					else:
						if weight is True:
							result[idx,0] = i
							result[idx,1] = self.edge_info[i].weight
						else:
							result[idx] = i
					idx += 1

			return result
		else:
			if data:
				result = numpy.empty( (self.node_info[nidx].indegree + self.node_info[nidx].outdegree,2), dtype=numpy.object_)
			else:
				result = numpy.empty(self.node_info[nidx].indegree + self.node_info[nidx].outdegree, dtype=numpy.object_)

			idx = 0
			num_edges = self.node_info[nidx].indegree
			elist = self.node_info[nidx].inelist
			for i in range(num_edges):
				if data is True:
					if weight is True:
						result[idx,0] = elist[i]
						result[idx,1] = self.edge_data_(elist[i])
						result[idx,2] = self.edge_info[elist[i]].weight
					else:
						result[idx,0] = elist[i]
						result[idx,1] = self.edge_data_(elist[i])
				else:
					if weight is True:
						result[idx,0] = elist[i]
						result[idx,1] = self.edge_info[elist[i]].weight
					else:
						result[idx] = elist[i]
				idx += 1

			# loop over out edges - this is copied verbatim from out_edges_iter(...)
			num_edges = self.node_info[nidx].outdegree
			elist = self.node_info[nidx].outelist
			for i in range(num_edges):
				if data is True:
					if weight is True:
						result[idx,0] = elist[i]
						result[idx,1] = self.edge_data_(elist[i])
						result[idx,2] = self.edge_info[elist[i]].weight
					else:
						result[idx,0] = elist[i]
						result[idx,1] = self.edge_data_(elist[i])
				else:
					if weight is True:
						result[idx,0] = elist[i]
						result[idx,1] = self.edge_info[elist[i]].weight
					else:
						result[idx] = elist[i]
				idx += 1

			return result

	cpdef in_edges_iter(DiGraph self,nobj,data=False,weight=False):
		"""
		Return an iterator over the edges for which ``nobj`` is a target.

		By default, the iterator will yield each edge as the tuple ``(u,tgt)``,
		where ``u`` and ``nobj`` are (node object) endpoints of the edge.

		**Args**:

			* ``nobj``: the object identifying the node from which all edges iterated over terminate at.

			* ``data [=False]`` (boolean): if ``True``, then the iterator adds object associated with the edge
			 	into the tuple returned (e.g., ``(u,v,d)``).

			* ``weight [=False]`` (boolean): 	if ``True``, then the iterator adds the weight of the edge
				into the tuple returned (e.g., ``(u,v,w)`` or ``(u,v,data,w)`` depending on the value of the ``data`` argument).
		"""
		return NodeEdgeIterator(self,self.node_idx_lookup[nobj],ITER_INDEGREE,weight,data,True)

	cpdef in_edges_iter_(DiGraph self,int nidx,data=False,weight=False):
		"""
		Return an iterator over the edges for which ``nidx`` (a node index) is a source.

		By default, the iterator will yield each edge as the edge index.

		**Args**:

			* ``nidx`` (int): the index for the node that is the terminating node for edges yielded.

			* ``data [=False]`` (boolean): if ``True``, then the iterator returns a tuple containing the edge index
				and the data associated with the edge (e.g., ``(eidx,d)``).

			* ``weight [=False]`` (boolean): 	if ``True``, then the iterator 	returns a tuple containing the edge index
				and the weight of the edge (e.g., ``(eidx,w)`` or ``(eidx,d,w)`` depending on the value of the ``data`` argument).
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx

		return NodeEdgeIterator(self,nidx,ITER_INDEGREE,weight,data)

	cpdef in_edges(DiGraph self,nobj,data=False,weight=False):
		"""
		Return a list of edges in the graph for which ``nobj`` (a node object) is a target.

		By default, the list will contain each edge as the tuple ``(u,nobj)``,
		where ``u`` and ``nobj`` are (node object) endpoints of the edge.

		**Args**:

			* ``nobj``: the object identifying the node from which all edges iterated over terminate at.

			* ``data [=False]`` (boolean): if ``True``, then the data object associated with the edge
			 	is added into the tuple returned for each edge (e.g., ``(u,v,d)``).

			* ``weight [=False]`` (boolean): 	if ``True``, then the weight of the edge is added
				into the tuple returned for each edge (e.g., ``(u,v,w)`` or ``(u,v,data,w)`` depending on the
				value of the ``data`` argument).
		"""
		return list(self.in_edges_iter(nobj,data,weight))

	cpdef in_edges_(DiGraph self,int nidx,data=False,weight=False):
		"""
		Return a ``numpy.ndarray`` containing edges in the graph that terminate at node with index ``nidx``.

		By default, the return value is a 1D array, ``R``, where ``R[i]`` is an edge index.

		**Args**:

			* ``nidx`` (int): the index for the node at which the edges included in the array terminate.

			* ``data [=False]`` (boolean): if ``True``, then the array will no longer be a 1D array.  A separate column will be added
			 	such that ``R[i,0]`` is the edge index and ``R[i,1]`` is the data object associated with the edge.

			* ``weight [=False]`` (boolean): 	if ``True``, then the array will no longer be a 1D array.  A separate column will be added
				such that ``R[i,0]`` is the edge index and ``R[i,1]`` is the weight of the edge.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx

		cdef int i
		cdef int* elist
		cdef int indegree
		result = None
		elist = self.node_info[nidx].inelist
		indegree = self.node_info[nidx].indegree

		if data and weight:
			result = numpy.empty( (indegree,3), dtype=numpy.object_)

			for i in range(indegree):
				result[i,0] = elist[i]
				result[i,1] = self.edge_data_lookup.get(elist[i],None)
				result[i,2] = self.edge_info[elist[i]].weight
		elif data:
			result = numpy.empty( (indegree,2), dtype=numpy.object_)

			for i in range(indegree):
				result[i,0] = elist[i]
				result[i,1] = self.edge_data_lookup.get(elist[i],None)
		elif weight:
			result = numpy.empty( (indegree,2), dtype=numpy.object_)

			for i in range(indegree):
				result[i,0] = elist[i]
				result[i,1] = self.edge_info[elist[i]].weight
		else:
			result = numpy.empty(indegree, dtype=numpy.int)

			for i in range(indegree):
				result[i] = elist[i]

		return result

	cpdef out_edges_iter(DiGraph self,nobj,data=False,weight=False):
		"""
		Return an iterator over the edges for which ``nobj`` is a source.

		By default, the iterator will yield each edge as the tuple ``(nobj,v)``,
		where ``nobj`` and ``v`` are (node object) endpoints of the edge.

		**Args**:

			* ``nobj``: the object identifying the node from which all edges iterated over eminate from.

			* ``data [=False]`` (boolean): if ``True``, then the iterator adds object associated with the edge
			 	into the tuple returned (e.g., ``(u,v,d)``).

			* ``weight [=False]`` (boolean): 	if ``True``, then the iterator adds the weight of the edge
				into the tuple returned (e.g., ``(u,v,w)`` or ``(u,v,data,w)`` depending on the value of the ``data`` argument).
		"""
		return NodeEdgeIterator(self,self.node_idx_lookup[nobj],ITER_OUTDEGREE,weight,data,True)

	cpdef out_edges_iter_(DiGraph self,int nidx,data=False,weight=False):
		"""
		Return an iterator over the edges for which ``nidx`` (a node index) is a source.

		By default, the iterator will yield each edge as the edge index.

		**Args**:

			* ``nidx`` (int): the index for the node that is the originating node for edges yielded.

			* ``data [=False]`` (boolean): if ``True``, then the iterator returns a tuple containing the edge index
				and the data associated with the edge (e.g., ``(eidx,d)``).

			* ``weight [=False]`` (boolean): 	if ``True``, then the iterator 	returns a tuple containing the edge index
				and the weight of the edge (e.g., ``(eidx,w)`` or ``(eidx,d,w)`` depending on the value of the ``data`` argument).
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx

		return NodeEdgeIterator(self,nidx,ITER_OUTDEGREE,weight,data)

	cpdef out_edges(DiGraph self,nobj,data=False,weight=False):
		"""
		Return a list of edges in the graph for which ``nobj`` (a node object) is a source.

		By default, the list will contain each edge as the tuple ``(nobj,v)``,
		where ``nobj`` and ``v`` are (node object) endpoints of the edge.

		**Args**:

			* ``nobj``: the object identifying the node from which all edges iterated over originate at.

			* ``data [=False]`` (boolean): if ``True``, then the data object associated with the edge
			 	is added into the tuple returned for each edge (e.g., ``(u,v,d)``).

			* ``weight [=False]`` (boolean): 	if ``True``, then the weight of the edge is added
				into the tuple returned for each edge (e.g., ``(u,v,w)`` or ``(u,v,data,w)`` depending on the
				value of the ``data`` argument).
		"""
		return list(self.out_edges_iter(nobj,data,weight))

	cpdef out_edges_(DiGraph self,int nidx,data=False,weight=False):
		"""
		Return a ``numpy.ndarray`` containing edges in the graph that originate at node with index ``nidx``.

		By default, the return value is a 1D array, ``R``, where ``R[i]`` is an edge index.

		**Args**:

			* ``nidx`` (int): the index for the node at which the edges included in the array originate.

			* ``data [=False]`` (boolean): if ``True``, then the array will no longer be a 1D array.  A separate column will be added
			 	such that ``R[i,0]`` is the edge index and ``R[i,1]`` is the data object associated with the edge.

			* ``weight [=False]`` (boolean): 	if ``True``, then the array will no longer be a 1D array.  A separate column will be added
				such that ``R[i,0]`` is the edge index and ``R[i,1]`` is the weight of the edge.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx

		cdef int i
		cdef int* elist
		cdef int outdegree

		result = None
		elist = self.node_info[nidx].outelist
		outdegree = self.node_info[nidx].outdegree

		if data and weight:
			result = numpy.empty( (outdegree,3), dtype=numpy.object_)

			for i in range(outdegree):
				result[i,0] = elist[i]
				result[i,1] = self.edge_data_lookup.get(elist[i],None)
				result[i,2] = self.edge_info[elist[i]].weight
		elif data:
			result = numpy.empty( (outdegree,2), dtype=numpy.object_)

			for i in range(outdegree):
				result[i,0] = elist[i]
				result[i,1] = self.edge_data_lookup.get(elist[i],None)
		elif weight:
			result = numpy.empty( (outdegree,2), dtype=numpy.object_)

			for i in range(outdegree):
				result[i,0] = elist[i]
				result[i,1] = self.edge_info[elist[i]].weight
		else:
			result = numpy.empty(outdegree, dtype=numpy.int)

			for i in range(outdegree):
				result[i] = elist[i]

		return result

	cpdef grp_edges_iter(DiGraph self,nbunch,data=False,weight=False):
		"""
		Return an iterator over the edges of a group of nodes.

		By default, the iterator will return each edge as the tuple ``(u,v)``, where ``u`` and ``v`` are (node object) endpoints of the edge.

		**Args**:

			* ``nbunch``: an iterable (usually a list) that yields node objects.  These are
				the nodes whose incident edges the iterator will return.

			* ``data [=False]`` (boolean): if ``True``, then the iterator adds object associated with the edge
			 	into the tuple returned (e.g., ``(u,v,d)``).

			* ``weight [=False]`` (boolean): 	if ``True``, then the iterator adds the weight of the edge
				into the tuple returned (e.g., ``(u,v,w)`` or ``(u,v,data,w)`` depending on the value of the ``data`` argument).
		"""
		return SomeEdgeIterator(self,[self.node_idx_lookup[x] for x in nbunch],ITER_BOTH,weight,data,True)

	cpdef grp_in_edges_iter(DiGraph self,nbunch,data=False,weight=False):
		"""
		Return an iterator over the edges that terminate within a group of nodes.

		By default, the iterator will return each edge as the tuple ``(u,v)``, where ``u`` and ``v`` are (node object) endpoints of the edge.

		**Args**:

			* ``nbunch``: an iterable (usually a list) that yields node objects.  These are
				the nodes whose terminal edges the iterator will return.

			* ``data [=False]`` (boolean): if ``True``, then the iterator adds object associated with the edge
			 	into the tuple returned (e.g., ``(u,v,d)``).

			* ``weight [=False]`` (boolean): 	if ``True``, then the iterator adds the weight of the edge
				into the tuple returned (e.g., ``(u,v,w)`` or ``(u,v,data,w)`` depending on the value of the ``data`` argument).
		"""
		return SomeEdgeIterator(self,[self.node_idx_lookup[x] for x in nbunch],ITER_INDEGREE,weight,data,True)

	cpdef grp_out_edges_iter(DiGraph self,nbunch,data=False,weight=False):
		"""
		Return an iterator over the edges that originate within a group of nodes.

		By default, the iterator will return each edge as the tuple ``(u,v)``, where ``u`` and ``v`` are (node object) endpoints of the edge.

		**Args**:

			* ``nbunch``: an iterable (usually a list) that yields node objects.  These are
				the nodes whose originating edges the iterator will return.

			* ``data [=False]`` (boolean): if ``True``, then the iterator adds object associated with the edge
			 	into the tuple returned (e.g., ``(u,v,d)``).

			* ``weight [=False]`` (boolean): 	if ``True``, then the iterator adds the weight of the edge
				into the tuple returned (e.g., ``(u,v,w)`` or ``(u,v,data,w)`` depending on the value of the ``data`` argument).
		"""
		return SomeEdgeIterator(self,[self.node_idx_lookup[x] for x in nbunch],ITER_OUTDEGREE,weight,data,True)

	cpdef grp_edges_iter_(DiGraph self,nbunch,data=False,weight=False):
		"""
		Return an iterator over edges incident to some nodes in the graph.

		By default, the iterator will return each edge as the edge index.

		**Args**:

			* ``nbunch``: an iterable (usually a list) that yields node indices.  These are
				the nodes whose originating edges the iterator will return.

			* ``data [=False]`` (boolean): if ``True``, then the iterator returns a tuple containing the edge index
				and the data associated with the edge (e.g., ``(eidx,d)``).

			* ``weight [=False]`` (boolean): 	if ``True``, then the iterator 	returns a tuple containing the edge index
				and the weight of the edge (e.g., ``(eidx,w)`` or ``(eidx,d,w)`` depending on the value of the ``data`` argument).
		"""
		for nidx in nbunch:
			if nidx >= self.node_capacity or not self.node_info[nidx].exists:
				raise ZenException, 'Invalid node idx %d' % nidx

		return SomeEdgeIterator(self,nbunch,ITER_BOTH,weight,data)

	cpdef grp_in_edges_iter_(DiGraph self,nbunch,data=False,weight=False):
		"""
		Return an iterator over edges that terminate among some nodes in the graph.

		By default, the iterator will return each edge as the edge index.

		**Args**:

			* ``nbunch``: an iterable (usually a list) that yields node indices.  These are
				the nodes among whom the edges returned terminate.

			* ``data [=False]`` (boolean): if ``True``, then the iterator returns a tuple containing the edge index
				and the data associated with the edge (e.g., ``(eidx,d)``).

			* ``weight [=False]`` (boolean): 	if ``True``, then the iterator 	returns a tuple containing the edge index
				and the weight of the edge (e.g., ``(eidx,w)`` or ``(eidx,d,w)`` depending on the value of the ``data`` argument).
		"""
		for nidx in nbunch:
			if nidx >= self.node_capacity or not self.node_info[nidx].exists:
				raise ZenException, 'Invalid node idx %d' % nidx

		return SomeEdgeIterator(self,nbunch,ITER_INDEGREE,weight,data)

	cpdef grp_out_edges_iter_(DiGraph self,nbunch,data=False,weight=False):
		"""
		Return an iterator over edges that originate among some nodes in the graph.

		By default, the iterator will return each edge as the edge index.

		**Args**:

			* ``nbunch``: an iterable (usually a list) that yields node indices.  These are
				the nodes among whom the edges returned originate.

			* ``data [=False]`` (boolean): if ``True``, then the iterator returns a tuple containing the edge index
				and the data associated with the edge (e.g., ``(eidx,d)``).

			* ``weight [=False]`` (boolean): 	if ``True``, then the iterator 	returns a tuple containing the edge index
				and the weight of the edge (e.g., ``(eidx,w)`` or ``(eidx,d,w)`` depending on the value of the ``data`` argument).
		"""
		for nidx in nbunch:
			if nidx >= self.node_capacity or not self.node_info[nidx].exists:
				raise ZenException, 'Invalid node idx %d' % nidx

		return SomeEdgeIterator(self,nbunch,ITER_OUTDEGREE,weight,data)

	cpdef neighbors(DiGraph self,nobj,data=False):
		"""
		Return a list of a node's immediate neighbors.

		By default, the list will contain the node object for each immediate neighbor of ``nobj``.

		**Args**:

			* ``nobj``: this is the node object identifying the node whose neighbors to retrieve.

			* ``data [=False]`` (boolean): if ``True``, then a tuple is returned containing the node
			 	object and the data object associated with the node (e.g., ``(n,d)``).
		"""
		cdef int num_edges
		cdef int* elist
		cdef int rid
		cdef nidx = self.node_idx_lookup[nobj]

		visited_neighbors = set()

		result = []
		idx = 0

		# loop over in edges
		num_edges = self.node_info[nidx].indegree
		elist = self.node_info[nidx].inelist
		for i in range(num_edges):
			rid = self.edge_info[elist[i]].src

			if rid in visited_neighbors:
				continue

			if data is True:
				result.append( (self.node_obj_lookup[rid], self.node_data_lookup[rid]) )
			else:
				if rid not in self.node_obj_lookup:
					raise ZenException, 'No node lookup known for node %d' % rid
				result.append(self.node_obj_lookup[rid])
			visited_neighbors.add(rid)

			idx += 1

		# loop over out edges
		num_edges = self.node_info[nidx].outdegree
		elist = self.node_info[nidx].outelist
		for i in range(num_edges):
			rid = self.edge_info[elist[i]].tgt

			if rid in visited_neighbors:
				continue

			if data is True:
				result.append( (self.node_obj_lookup[rid], self.node_data_lookup[rid]) )
			else:
				if rid not in self.node_obj_lookup:
					raise ZenException, 'No node lookup known for node %d' % rid
				result.append(self.node_obj_lookup[rid])
			visited_neighbors.add(rid)

			idx += 1

		return result

	cpdef neighbors_(DiGraph self,int nidx,obj=False,data=False):
		"""
		Return an ``numpy.ndarray`` containing a node's immediate neighbors.

		By default, the return value will be a 1D array containing the node index for each immediate neighbor of ``nidx``.

		**Args**:

			* ``nidx``: this is the node index identifying the node whose neighbors to retrieve.

			* ``obj [=False]`` (boolean): if ``True``, then a 2D array, ``R`` is returned in which ``R[i,0]`` is the index
				of the neighbor and ``R[i,1]`` is the node object associated with it.

			* ``data [=False]`` (boolean): if ``True``, then a 2D array, ``R``, is returned with the final column containing the
				data object associated with the neighbor (e.g., ``R[i,0]`` is the index	of the neighbor and ``R[i,1]`` or ``R[i,2]``
				is the data object, depending on the value of the ``nobj`` argument).
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx

		cdef int num_edges
		cdef int* elist
		cdef int rid

		visited_neighbors = set()

		result = None
		if obj or data:
			ndim = 1 if not data and not obj else (2 if not data or not obj else 3)
			result = numpy.empty( (self.node_info[nidx].indegree + self.node_info[nidx].outdegree,ndim), dtype=numpy.object_)
		else:
			result = numpy.empty( self.node_info[nidx].indegree + self.node_info[nidx].outdegree, dtype=numpy.object_)

		idx = 0

		# loop over in edges
		num_edges = self.node_info[nidx].indegree
		elist = self.node_info[nidx].inelist
		for i in range(num_edges):
			rid = self.edge_info[elist[i]].src

			if rid in visited_neighbors:
				continue

			visited_neighbors.add(rid)

			if not obj and not data:
				result[idx] = rid
			elif obj and not data:
				result[idx,0] = rid
				if rid in self.node_obj_lookup[rid]:
					result[idx,1] = self.node_obj_lookup[rid]
				else:
					result[idx,1] = None
			elif not obj and data:
				result[idx,0] = rid
				if rid in self.node_data_lookup[rid]:
					result[idx,1] = self.node_data_lookup[rid]
				else:
					result[idx,1] = None
			else:
				result[idx,0] = rid
				if rid in self.node_obj_lookup[rid]:
					result[idx,1] = self.node_obj_lookup[rid]
				else:
					result[idx,1] = None
				if rid in self.node_data_lookup[rid]:
					result[idx,2] = self.node_data_lookup[rid]
				else:
					result[idx,2] = None

			idx += 1

		# loop over out edges
		num_edges = self.node_info[nidx].outdegree
		elist = self.node_info[nidx].outelist
		for i in range(num_edges):
			rid = self.edge_info[elist[i]].tgt

			if rid in visited_neighbors:
				continue

			visited_neighbors.add(rid)

			if not obj and not data:
				result[idx] = rid
			elif obj and not data:
				result[idx,0] = rid
				if rid in self.node_obj_lookup[rid]:
					result[idx,1] = self.node_obj_lookup[rid]
				else:
					result[idx,1] = None
			elif not obj and data:
				result[idx,0] = rid
				if rid in self.node_data_lookup[rid]:
					result[idx,1] = self.node_data_lookup[rid]
				else:
					result[idx,1] = None
			else:
				result[idx,0] = rid
				if rid in self.node_obj_lookup[rid]:
					result[idx,1] = self.node_obj_lookup[rid]
				else:
					result[idx,1] = None
				if rid in self.node_data_lookup[rid]:
					result[idx,2] = self.node_data_lookup[rid]
				else:
					result[idx,2] = None
			idx += 1

		if data:
			result.resize( (idx,2) )
		else:
			result.resize(idx)

		return result

	cpdef neighbors_iter(DiGraph self,nobj,data=False):
		"""
		Return an iterator over a node's immediate neighbors.

		By default, the iterator will yield the node object for each immediate neighbor of ``nobj``.

		**Args**:

			* ``nobj``: this is the node object identifying the node whose neighbors to iterate over.

			* ``data [=False]`` (boolean): if ``True``, then a tuple is returned (rather than a node object)
				containing the node	object and the data object associated with the node (e.g., ``(n,d)``).
		"""
		return NeighborIterator(self,self.node_idx_lookup[nobj],ITER_BOTH,False,data,True)

	cpdef neighbors_iter_(DiGraph self,int nidx,obj=False,data=False):
		"""
		Return an iterator over a node's immediate neighbors.

		By default, the iterator will yield the node index for each immediate neighbor of ``nidx``.

		**Args**:

			* ``nidx``: this is the node index identifying the node whose neighbors to iterate over.

			* ``obj [=False]`` (boolean): if ``True``, then a tuple is returned (rather than a node index)
				containing the node	index and the node object associated with the node (e.g., ``(nidx,n)``).

			* ``data [=False]`` (boolean): if ``True``, then a tuple is returned (rather than a node index)
				containing the node	index and the data object associated with the node (e.g., ``(nidx,d)`` or ``(nidx,n,d)``
				depending on the value of the ``nobj`` argument).
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx

		return NeighborIterator(self,nidx,ITER_BOTH,obj,data,False)

	cpdef in_neighbors_iter(DiGraph self,nobj,data=False):
		"""
		Return an iterator over a node's immediate in-bound neighbors.

		By default, the iterator will yield the node object for each immediate out-bound neighbor of ``nobj``.

		**Args**:

			* ``nobj``: this is the node object identifying the node whose in-bound neighbors to iterate over.

			* ``data [=False]`` (boolean): if ``True``, then a tuple is returned (rather than a node object)
				containing the node	object and the data object associated with the node (e.g., ``(n,d)``).
		"""
		return NeighborIterator(self,self.node_idx_lookup[nobj],ITER_INDEGREE,False,data,True)

	cpdef in_neighbors_iter_(DiGraph self,int nidx,obj=False,data=False):
		"""
		Return an iterator over a node's immediate in-bound neighbors.

		By default, the iterator will yield the node index for each immediate in-bound neighbor of ``nidx``.

		**Args**:

			* ``nidx``: this is the node index identifying the node whose in-bound neighbors to iterate over.

			* ``obj [=False]`` (boolean): if ``True``, then a tuple is returned (rather than a node index)
				containing the node	index and the node object associated with the node (e.g., ``(nidx,n)``).

			* ``data [=False]`` (boolean): if ``True``, then a tuple is returned (rather than a node index)
				containing the node	index and the data object associated with the node (e.g., ``(nidx,d)`` or ``(nidx,n,d)``
				depending on the value of the ``nobj`` argument).
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx

		return NeighborIterator(self,nidx,ITER_INDEGREE,obj,data,False)

	cpdef in_neighbors(DiGraph self,nobj,data=False):
		"""
		Return a list of a node's immediate in-bound neighbors.

		By default, the list will contain the node object for each immediate in-bound neighbor of ``nobj``.

		**Args**:

			* ``nobj``: this is the node object identifying the node whose in-bound neighbors to retrieve.

			* ``data [=False]`` (boolean): if ``True``, then a tuple is returned containing the node
			 	object and the data object associated with the node (e.g., ``(n,d)``).
		"""
		return list(self.in_neighbors_iter(nobj,data))

	cpdef in_neighbors_(DiGraph self,int nidx,obj=False,data=False):
		"""
		Return an ``numpy.ndarray`` containing a node's immediate in-bound neighbors.

		By default, the return value will be a 1D array containing the node index for each immediate in-bound neighbor of ``nidx``.

		**Args**:

			* ``nidx``: this is the node index identifying the node whose in-bound neighbors to retrieve.

			* ``obj [=False]`` (boolean): if ``True``, then a 2D array, ``R`` is returned in which ``R[i,0]`` is the index
				of the neighbor and ``R[i,1]`` is the node object associated with it.

			* ``data [=False]`` (boolean): if ``True``, then a 2D array, ``R``, is returned with the final column containing the
				data object associated with the neighbor (e.g., ``R[i,0]`` is the index	of the neighbor and ``R[i,1]`` or ``R[i,2]``
				is the data object, depending on the value of the ``nobj`` argument).
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx

		cdef int i
		cdef int* elist
		cdef int indegree
		result = None
		elist = self.node_info[nidx].inelist
		indegree = self.node_info[nidx].indegree

		if obj and not data:
			result = numpy.empty( (indegree,2), dtype=numpy.object_)

			for i in range(indegree):
				result[i,0] = self.edge_info[elist[i]].src
				result[i,1] = self.node_obj_lookup[self.edge_info[elist[i]].src]
		elif data and not obj:
			result = numpy.empty( (indegree,2), dtype=numpy.object_)

			for i in range(indegree):
				result[i,0] = self.edge_info[elist[i]].src
				result[i,1] = self.node_data_lookup[self.edge_info[elist[i]].src]
		elif data and obj:
			result = numpy.empty( (indegree,3), dtype=numpy.object_)

			for i in range(indegree):
				result[i,0] = self.edge_info[elist[i]].src
				result[i,1] = self.node_obj_lookup[self.edge_info[elist[i]].src]
				result[i,2] = self.node_data_lookup[self.edge_info[elist[i]].src]
		else:
			result = numpy.empty(indegree, dtype=numpy.int)

			for i in range(indegree):
				result[i] = self.edge_info[elist[i]].src

		return result

	cpdef out_neighbors_iter(DiGraph self,nobj,data=False):
		"""
		Return an iterator over a node's immediate out-bound neighbors.

		By default, the iterator will yield the node object for each immediate out-bound neighbor of ``nobj``.

		**Args**:

			* ``nobj``: this is the node object identifying the node whose out-bound neighbors to iterate over.

			* ``data [=False]`` (boolean): if ``True``, then a tuple is returned (rather than a node object)
				containing the node	object and the data object associated with the node (e.g., ``(n,d)``).
		"""
		return NeighborIterator(self,self.node_idx_lookup[nobj],ITER_OUTDEGREE,False,data,True)

	cpdef out_neighbors_iter_(DiGraph self,int nidx,obj=False,data=False):
		"""
		Return an iterator over a node's immediate out-bound neighbors.

		By default, the iterator will yield the node index for each immediate out-bound neighbor of ``nidx``.

		**Args**:

			* ``nidx``: this is the node index identifying the node whose out-bound neighbors to iterate over.

			* ``obj [=False]`` (boolean): if ``True``, then a tuple is returned (rather than a node index)
				containing the node	index and the node object associated with the node (e.g., ``(nidx,n)``).

			* ``data [=False]`` (boolean): if ``True``, then a tuple is returned (rather than a node index)
				containing the node	index and the data object associated with the node (e.g., ``(nidx,d)`` or ``(nidx,n,d)``
				depending on the value of the ``nobj`` argument).
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx

		return NeighborIterator(self,nidx,ITER_OUTDEGREE,obj,data,False)

	cpdef out_neighbors(DiGraph self,nobj,data=False):
		"""
		Return a list of a node's immediate out-bound neighbors.

		By default, the list will contain the node object for each immediate out-bound neighbor of ``nobj``.

		**Args**:

			* ``nobj``: this is the node object identifying the node whose out-bound neighbors to retrieve.

			* ``data [=False]`` (boolean): if ``True``, then a tuple is returned containing the node
			 	object and the data object associated with the node (e.g., ``(n,d)``).
		"""
		return list(self.out_neighbors_iter(nobj,data))

	cpdef out_neighbors_(DiGraph self,int nidx,obj=False,data=False):
		"""
		Return an ``numpy.ndarray`` containing a node's immediate out-bound neighbors.

		By default, the return value will be a 1D array containing the node index for each immediate out-bound neighbor of ``nidx``.

		**Args**:

			* ``nidx``: this is the node index identifying the node whose out-bound neighbors to retrieve.

			* ``obj [=False]`` (boolean): if ``True``, then a 2D array, ``R`` is returned in which ``R[i,0]`` is the index
				of the neighbor and ``R[i,1]`` is the node object associated with it.

			* ``data [=False]`` (boolean): if ``True``, then a 2D array, ``R``, is returned with the final column containing the
				data object associated with the neighbor (e.g., ``R[i,0]`` is the index	of the neighbor and ``R[i,1]`` or ``R[i,2]``
				is the data object, depending on the value of the ``nobj`` argument).
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx

		cdef int i
		cdef int* elist
		cdef int outdegree
		result = None
		elist = self.node_info[nidx].outelist
		outdegree = self.node_info[nidx].outdegree

		if obj and not data:
			result = numpy.empty( (outdegree,2), dtype=numpy.object_)

			for i in range(outdegree):
				result[i,0] = self.edge_info[elist[i]].tgt
				result[i,1] = self.node_obj_lookup[self.edge_info[elist[i]].tgt]
		elif data and not obj:
			result = numpy.empty( (outdegree,2), dtype=numpy.object_)

			for i in range(outdegree):
				result[i,0] = self.edge_info[elist[i]].tgt
				result[i,1] = self.node_data_lookup[self.edge_info[elist[i]].tgt]
		elif data and obj:
			result = numpy.empty( (outdegree,3), dtype=numpy.object_)

			for i in range(outdegree):
				result[i,0] = self.edge_info[elist[i]].tgt
				result[i,1] = self.node_obj_lookup[self.edge_info[elist[i]].tgt]
				result[i,2] = self.node_data_lookup[self.edge_info[elist[i]].tgt]
		else:
			result = numpy.empty(outdegree, dtype=numpy.int)

			for i in range(outdegree):
				result[i] = self.edge_info[elist[i]].tgt

		return result

	cpdef grp_neighbors_iter(DiGraph self,nbunch,data=False):
		"""
		Return an iterator over a group of nodes' immediate neighbors.

		By default, the iterator will yield the node object for each immediate neighbor of nodes in ``nbunch``.

		**Args**:

			* ``nbunch``: an iterable providing the node object over whose neighbors to iterate.

			* ``data [=False]`` (boolean): if ``True``, then a tuple is yielded (rather than a node object)
				containing the node	object and the data object associated with the node (e.g., ``(n,d)``).
		"""
		return SomeNeighborIterator(self,[self.node_idx_lookup[x] for x in nbunch],ITER_BOTH,False,data,True)

	cpdef grp_in_neighbors_iter(DiGraph self,nbunch,data=False):
		"""
		Return an iterator over a group of nodes' immediate in-bound neighbors.

		By default, the iterator will yield the node object for each immediate in-bound neighbor of nodes in ``nbunch``.

		**Args**:

			* ``nbunch``: an iterable providing the node object over whose in-bound neighbors to iterate.

			* ``data [=False]`` (boolean): if ``True``, then a tuple is yielded (rather than a node object)
				containing the node	object and the data object associated with the node (e.g., ``(n,d)``).
		"""
		return SomeNeighborIterator(self,[self.node_idx_lookup[x] for x in nbunch],ITER_INDEGREE,False,data,True)

	cpdef grp_out_neighbors_iter(DiGraph self,nbunch,data=False):
		"""
		Return an iterator over a group of nodes' immediate out-bound neighbors.

		By default, the iterator will yield the node object for each immediate out-bound neighbor of nodes in ``nbunch``.

		**Args**:

			* ``nbunch``: an iterable providing the node object over whose out-bound neighbors to iterate.

			* ``data [=False]`` (boolean): if ``True``, then a tuple is yielded (rather than a node object)
				containing the node	object and the data object associated with the node (e.g., ``(n,d)``).
		"""
		return SomeNeighborIterator(self,[self.node_idx_lookup[x] for x in nbunch],ITER_OUTDEGREE,False,data,True)

	cpdef grp_neighbors_iter_(DiGraph self,nbunch,obj=False,data=False):
		"""
		Return an iterator over a group of nodes' immediate neighbors.

		By default, the iterator will yield the node index for each immediate neighbor of nodes in the iterable ``nbunch``.

		**Args**:

			* ``nbunch``: an iterable providing the node indices over whose neighbors to iterate.

			* ``obj [=False]`` (boolean): if ``True``, then a tuple is yielded (rather than a node index)
				containing the node	index and the node object associated with the node (e.g., ``(nidx,n)``).

			* ``data [=False]`` (boolean): if ``True``, then a tuple is yielded (rather than a node index)
				containing the node	index and the data object associated with the node (e.g., ``(nidx,d)`` or ``(nidx,n,d)``
				depending on the value of the ``nobj`` argument).
		"""
		for nidx in nbunch:
			if nidx >= self.node_capacity or not self.node_info[nidx].exists:
				raise ZenException, 'Invalid node idx %d' % nidx

		return SomeNeighborIterator(self,nbunch,ITER_BOTH,obj,data,False)

	cpdef grp_in_neighbors_iter_(DiGraph self,nbunch,obj=False,data=False):
		"""
		Return an iterator over a group of nodes' immediate in-bound neighbors.

		By default, the iterator will yield the node index for each immediate in-bound neighbor of nodes in the iterable ``nbunch``.

		**Args**:

			* ``nbunch``: an iterable providing the node indices over whose in-bound neighbors to iterate.

			* ``obj [=False]`` (boolean): if ``True``, then a tuple is yielded (rather than a node index)
				containing the node	index and the node object associated with the node (e.g., ``(nidx,n)``).

			* ``data [=False]`` (boolean): if ``True``, then a tuple is yielded (rather than a node index)
				containing the node	index and the data object associated with the node (e.g., ``(nidx,d)`` or ``(nidx,n,d)``
				depending on the value of the ``nobj`` argument).
		"""
		for nidx in nbunch:
			if nidx >= self.node_capacity or not self.node_info[nidx].exists:
				raise ZenException, 'Invalid node idx %d' % nidx

		return SomeNeighborIterator(self,nbunch,ITER_INDEGREE,obj,data,False)

	cpdef grp_out_neighbors_iter_(DiGraph self,nbunch,obj=False,data=False):
		"""
		Return an iterator over a group of nodes' immediate out-bound neighbors.

		By default, the iterator will yield the node index for each immediate out-bound neighbor of nodes in the iterable ``nbunch``.

		**Args**:

			* ``nbunch``: an iterable providing the node indices over whose out-bound neighbors to iterate.

			* ``obj [=False]`` (boolean): if ``True``, then a tuple is yielded (rather than a node index)
				containing the node	index and the node object associated with the node (e.g., ``(nidx,n)``).

			* ``data [=False]`` (boolean): if ``True``, then a tuple is yielded (rather than a node index)
				containing the node	index and the data object associated with the node (e.g., ``(nidx,d)`` or ``(nidx,n,d)``
				depending on the value of the ``nobj`` argument).
		"""
		for nidx in nbunch:
			if nidx >= self.node_capacity or not self.node_info[nidx].exists:
				raise ZenException, 'Invalid node idx %d' % nidx

		return SomeNeighborIterator(self,nbunch,ITER_OUTDEGREE,obj,data,False)

cdef class NodeIterator:
	cdef bint data
	cdef DiGraph graph
	cdef int idx
	cdef int node_count
	cdef bint nobj
	cdef bint obj
	cdef long init_num_changes

	def __cinit__(NodeIterator self,DiGraph graph,bint obj,bint data,bint nobj):
		self.init_num_changes = graph.num_changes
		self.data = data
		self.graph = graph
		self.idx = 0
		self.node_count = 0
		self.nobj = nobj
		self.obj = obj

	def __next__(NodeIterator self):
		if self.init_num_changes != self.graph.num_changes:
			raise GraphChangedException()

		if self.node_count >= self.graph.num_nodes:
			raise StopIteration()

		cdef int idx = self.idx

		while idx < self.graph.node_capacity and not self.graph.node_info[idx].exists:
			idx += 1

		if idx >= self.graph.node_capacity:
			self.node_count = self.graph.num_nodes
			raise StopIteration()

		self.node_count += 1
		self.idx = idx + 1

		if self.nobj:
			if idx not in self.graph.node_obj_lookup:
				raise ZenException, 'Node (idx=%d) missing object' % idx
			obj = self.graph.node_obj_lookup[idx]
			if self.data:
				val = None
				if idx in self.graph.node_data_lookup:
					val = self.graph.node_data_lookup[idx]
				return obj,val
			else:
				return obj
		else:
			obj = None
			data = None
			if self.obj and idx in self.graph.node_obj_lookup:
				obj = self.graph.node_obj_lookup[idx]
			if self.data and idx in self.graph.node_data_lookup:
				data = self.graph.node_data_lookup[idx]

			if not self.obj and not self.data:
				return idx
			elif self.obj and not self.data:
				return idx,obj
			elif not self.obj and self.data:
				return idx,data
			else:
				return idx,obj,data

	def __iter__(NodeIterator self):
		return self

cdef class AllEdgeIterator:
	cdef bint data
	cdef bint weight
	cdef DiGraph graph
	cdef int idx
	cdef bint endpoints
	cdef long init_num_changes

	def __cinit__(AllEdgeIterator self,DiGraph graph,weight=False,data=False,endpoints=False):
		self.init_num_changes = graph.num_changes
		self.graph = graph
		self.data = data
		self.weight = weight
		self.idx = 0
		self.endpoints = endpoints

	def __next__(AllEdgeIterator self):
		if self.init_num_changes != self.graph.num_changes:
			raise GraphChangedException()

		cdef int idx = self.idx

		if idx == self.graph.next_edge_idx:
			raise StopIteration()

		while idx < self.graph.next_edge_idx and not self.graph.edge_info[idx].exists:
			idx += 1

		if idx >= self.graph.next_edge_idx:
			self.idx = idx
			raise StopIteration()

		self.idx = idx + 1
		if self.weight and self.data:
			val = None
			if idx in self.graph.edge_data_lookup:
				val = self.graph.edge_data_lookup[idx]

			if not self.endpoints:
				return idx, val, self.graph.edge_info[idx].weight
			else:
				if self.graph.edge_info[idx].src not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Source node (idx=%d) does not have an object' % self.graph.edge_info[idx].src
				elif self.graph.edge_info[idx].tgt not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Target node (idx=%d) does not have an object' % self.graph.edge_info[idx].tgt

				return self.graph.node_obj_lookup[self.graph.edge_info[idx].src], self.graph.node_obj_lookup[self.graph.edge_info[idx].tgt], val, self.graph.edge_info[idx].weight
		elif self.weight:
			if not self.endpoints:
				return idx, self.graph.edge_info[idx].weight
			else:
				if self.graph.edge_info[idx].src not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Source node (idx=%d) does not have an object' % self.graph.edge_info[idx].src
				elif self.graph.edge_info[idx].tgt not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Target node (idx=%d) does not have an object' % self.graph.edge_info[idx].tgt

				return self.graph.node_obj_lookup[self.graph.edge_info[idx].src], self.graph.node_obj_lookup[self.graph.edge_info[idx].tgt], self.graph.edge_info[idx].weight
		elif self.data:
			val = None
			if idx in self.graph.edge_data_lookup:
				val = self.graph.edge_data_lookup[idx]
			if not self.endpoints:
				return idx, val
			else:
				if self.graph.edge_info[idx].src not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Source node (idx=%d) does not have an object' % self.graph.edge_info[idx].src
				elif self.graph.edge_info[idx].tgt not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Target node (idx=%d) does not have an object' % self.graph.edge_info[idx].tgt

				return self.graph.node_obj_lookup[self.graph.edge_info[idx].src], self.graph.node_obj_lookup[self.graph.edge_info[idx].tgt], val
		else:
			if not self.endpoints:
				return idx
			else:
				if self.graph.edge_info[idx].src not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Source node (idx=%d) does not have an object' % self.graph.edge_info[idx].src
				elif self.graph.edge_info[idx].tgt not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Target node (idx=%d) does not have an object' % self.graph.edge_info[idx].tgt

				return self.graph.node_obj_lookup[self.graph.edge_info[idx].src], self.graph.node_obj_lookup[self.graph.edge_info[idx].tgt]

	def __iter__(AllEdgeIterator self):
		return self

cdef int ITER_BOTH = 0
cdef int ITER_INDEGREE = 1
cdef int ITER_OUTDEGREE = 2

cdef class NodeEdgeIterator:
	cdef bint data
	cdef bint weight
	cdef DiGraph graph
	cdef int nidx
	cdef int deg
	cdef int idx
	cdef int which_degree
	cdef bint endpoints
	cdef long init_num_changes

	def __cinit__(NodeEdgeIterator self,DiGraph graph,nidx,which_degree,weight=False,data=False,endpoints=False):
		self.init_num_changes = graph.num_changes
		self.graph = graph
		self.nidx = nidx
		self.data = data
		self.weight = weight
		self.idx = 0
		self.which_degree = which_degree
		self.endpoints = endpoints

		self.deg = 0
		if self.which_degree == ITER_BOTH:
			if self.graph.node_info[nidx].indegree > 0:
				self.deg = 1
			elif self.graph.node_info[nidx].outdegree > 0:
				self.deg = 2
		elif self.which_degree == ITER_INDEGREE and self.graph.node_info[nidx].indegree > 0:
			self.deg = 1
		elif self.which_degree == ITER_OUTDEGREE and self.graph.node_info[nidx].outdegree > 0:
			self.deg = 2

	def __next__(NodeEdgeIterator self):
		if self.init_num_changes != self.graph.num_changes:
			raise GraphChangedException()

		cdef int idx = self.idx
		cdef int* elist

		if self.deg == 0:
			raise StopIteration()
		elif self.deg == 1:
			num_edges = self.graph.node_info[self.nidx].indegree
			elist = self.graph.node_info[self.nidx].inelist

			if (idx + 1) == num_edges:
				self.idx = 0
				if self.graph.node_info[self.nidx].outdegree > 0 and self.which_degree == ITER_BOTH:
					self.deg = 2
				else:
					self.deg = 0
			else:
				self.idx = idx + 1

			if self.weight and self.data:
				val = None
				if elist[idx] in self.graph.edge_data_lookup:
					val = self.graph.edge_data_lookup[elist[idx]]

				idx = elist[idx]
				if not self.endpoints:
					return idx, val, self.graph.edge_info[idx].weight
				else:
					if self.graph.edge_info[idx].src not in self.graph.node_obj_lookup:
						raise ZenException, 'Missing endpoint: Source node (idx=%d) does not have an object' % self.graph.edge_info[idx].src
					elif self.graph.edge_info[idx].tgt not in self.graph.node_obj_lookup:
						raise ZenException, 'Missing endpoint: Target node (idx=%d) does not have an object' % self.graph.edge_info[idx].tgt

					return self.graph.node_obj_lookup[self.graph.edge_info[idx].src], self.graph.node_obj_lookup[self.graph.edge_info[idx].tgt], val, self.graph.edge_info[idx].weight
			elif self.weight:
				idx = elist[idx]
				if not self.endpoints:
					return idx, self.graph.edge_info[idx].weight
				else:
					if self.graph.edge_info[idx].src not in self.graph.node_obj_lookup:
						raise ZenException, 'Missing endpoint: Source node (idx=%d) does not have an object' % self.graph.edge_info[idx].src
					elif self.graph.edge_info[idx].tgt not in self.graph.node_obj_lookup:
						raise ZenException, 'Missing endpoint: Target node (idx=%d) does not have an object' % self.graph.edge_info[idx].tgt

					return self.graph.node_obj_lookup[self.graph.edge_info[idx].src], self.graph.node_obj_lookup[self.graph.edge_info[idx].tgt], self.graph.edge_info[idx].weight
			elif self.data:
				val = None
				if elist[idx] in self.graph.edge_data_lookup:
					val = self.graph.edge_data_lookup[elist[idx]]
				if not self.endpoints:
					return elist[idx], val
				else:
					idx = elist[idx]
					if self.graph.edge_info[idx].src not in self.graph.node_obj_lookup:
						raise ZenException, 'Missing endpoint: Source node (idx=%d) does not have an object' % self.graph.edge_info[idx].src
					elif self.graph.edge_info[idx].tgt not in self.graph.node_obj_lookup:
						raise ZenException, 'Missing endpoint: Target node (idx=%d) does not have an object' % self.graph.edge_info[idx].tgt

					return self.graph.node_obj_lookup[self.graph.edge_info[idx].src], self.graph.node_obj_lookup[self.graph.edge_info[idx].tgt], val
			else:
				if not self.endpoints:
					return elist[idx]
				else:
					idx = elist[idx]
					if self.graph.edge_info[idx].src not in self.graph.node_obj_lookup:
						raise ZenException, 'Missing endpoint: Source node (idx=%d) does not have an object' % self.graph.edge_info[idx].src
					elif self.graph.edge_info[idx].tgt not in self.graph.node_obj_lookup:
						raise ZenException, 'Missing endpoint: Target node (idx=%d) does not have an object' % self.graph.edge_info[idx].tgt

					return self.graph.node_obj_lookup[self.graph.edge_info[idx].src], self.graph.node_obj_lookup[self.graph.edge_info[idx].tgt]

		elif self.deg == 2:
			num_edges = self.graph.node_info[self.nidx].outdegree
			elist = self.graph.node_info[self.nidx].outelist

			if (idx + 1) == num_edges:
				self.idx = 0
				self.deg = 0
			else:
				self.idx = idx + 1

			if self.weight and self.data:
				val = None
				if elist[idx] in self.graph.edge_data_lookup:
					val = self.graph.edge_data_lookup[elist[idx]]

				idx = elist[idx]
				if not self.endpoints:
					return idx, val, self.graph.edge_info[idx].weight
				else:
					if self.graph.edge_info[idx].src not in self.graph.node_obj_lookup:
						raise ZenException, 'Missing endpoint: Source node (idx=%d) does not have an object' % self.graph.edge_info[idx].src
					elif self.graph.edge_info[idx].tgt not in self.graph.node_obj_lookup:
						raise ZenException, 'Missing endpoint: Target node (idx=%d) does not have an object' % self.graph.edge_info[idx].tgt

					return self.graph.node_obj_lookup[self.graph.edge_info[idx].src], self.graph.node_obj_lookup[self.graph.edge_info[idx].tgt], val, self.graph.edge_info[idx].weight
			elif self.weight:
				idx = elist[idx]
				if not self.endpoints:
					return idx, self.graph.edge_info[idx].weight
				else:
					if self.graph.edge_info[idx].src not in self.graph.node_obj_lookup:
						raise ZenException, 'Missing endpoint: Source node (idx=%d) does not have an object' % self.graph.edge_info[idx].src
					elif self.graph.edge_info[idx].tgt not in self.graph.node_obj_lookup:
						raise ZenException, 'Missing endpoint: Target node (idx=%d) does not have an object' % self.graph.edge_info[idx].tgt

					return self.graph.node_obj_lookup[self.graph.edge_info[idx].src], self.graph.node_obj_lookup[self.graph.edge_info[idx].tgt], self.graph.edge_info[idx].weight
			elif self.data:
				val = None
				if elist[idx] in self.graph.edge_data_lookup:
					val = self.graph.edge_data_lookup[elist[idx]]
				if not self.endpoints:
					return elist[idx], val
				else:
					idx = elist[idx]
					if self.graph.edge_info[idx].src not in self.graph.node_obj_lookup:
						raise ZenException, 'Missing endpoint: Source node (idx=%d) does not have an object' % self.graph.edge_info[idx].src
					elif self.graph.edge_info[idx].tgt not in self.graph.node_obj_lookup:
						raise ZenException, 'Missing endpoint: Target node (idx=%d) does not have an object' % self.graph.edge_info[idx].tgt

					return self.graph.node_obj_lookup[self.graph.edge_info[idx].src], self.graph.node_obj_lookup[self.graph.edge_info[idx].tgt], val
			else:
				if not self.endpoints:
					return elist[idx]
				else:
					idx = elist[idx]
					if self.graph.edge_info[idx].src not in self.graph.node_obj_lookup:
						raise ZenException, 'Missing endpoint: Source node (idx=%d) does not have an object' % self.graph.edge_info[idx].src
					elif self.graph.edge_info[idx].tgt not in self.graph.node_obj_lookup:
						raise ZenException, 'Missing endpoint: Target node (idx=%d) does not have an object' % self.graph.edge_info[idx].tgt

					return self.graph.node_obj_lookup[self.graph.edge_info[idx].src], self.graph.node_obj_lookup[self.graph.edge_info[idx].tgt]

	def __iter__(NodeEdgeIterator self):
		return self

cdef class SomeEdgeIterator:
	cdef bint data
	cdef bint weight
	cdef DiGraph graph
	cdef int which_degree
	cdef touched_edges
	cdef nbunch_iter
	cdef edge_iter
	cdef bint endpoints
	cdef long init_num_changes

	def __cinit__(SomeEdgeIterator self,DiGraph graph,nbunch,which_degree,weight=False,data=False,endpoints=False):
		self.init_num_changes = graph.num_changes
		self.graph = graph
		self.nbunch_iter = iter(nbunch)
		self.data = data
		self.weight = weight
		self.which_degree = which_degree
		self.edge_iter = None
		self.touched_edges = set()
		self.endpoints = endpoints

		# setup the first iterator
		if len(nbunch) > 0:
			curr_nidx = self.nbunch_iter.next()
			self.edge_iter = NodeEdgeIterator(self.graph,curr_nidx,self.which_degree,self.weight,self.data)
		else:
			self.edge_iter = None

	def __next__(SomeEdgeIterator self):
		if self.init_num_changes != self.graph.num_changes:
			raise GraphChangedException()

		while True:
			if self.edge_iter is None:
				raise StopIteration
			else:
				try:
					result = self.edge_iter.next()
					if self.weight and self.data:
						if result[0] in self.touched_edges:
							continue
						self.touched_edges.add(result[0])

						if not self.endpoints:
							return result
						else:
							if self.graph.edge_info[result[0]].src not in self.graph.node_obj_lookup:
								raise ZenException, 'Missing endpoint: Source node (idx=%d) does not have an object' % self.graph.edge_info[result[0]].src
							elif self.graph.edge_info[result[0]].tgt not in self.graph.node_obj_lookup:
								raise ZenException, 'Missing endpoint: Target node (idx=%d) does not have an object' % self.graph.edge_info[result[0]].tgt

							return self.graph.node_obj_lookup[self.graph.edge_info[result[0]].src], self.graph.node_obj_lookup[self.graph.edge_info[result[0]].tgt], result[1], result[2]
					elif self.weight or self.data:
						if result[0] in self.touched_edges:
							continue
						self.touched_edges.add(result[0])

						if not self.endpoints:
							return result
						else:
							if self.graph.edge_info[result[0]].src not in self.graph.node_obj_lookup:
								raise ZenException, 'Missing endpoint: Source node (idx=%d) does not have an object' % self.graph.edge_info[result[0]].src
							elif self.graph.edge_info[result[0]].tgt not in self.graph.node_obj_lookup:
								raise ZenException, 'Missing endpoint: Target node (idx=%d) does not have an object' % self.graph.edge_info[result[0]].tgt

							return self.graph.node_obj_lookup[self.graph.edge_info[result[0]].src], self.graph.node_obj_lookup[self.graph.edge_info[result[0]].tgt], result[1]
					else:
						if result in self.touched_edges:
							continue
						self.touched_edges.add(result)

						if not self.endpoints:
							return result
						else:
							if self.graph.edge_info[result].src not in self.graph.node_obj_lookup:
								raise ZenException, 'Missing endpoint: Source node (idx=%d) does not have an object' % self.graph.edge_info[result].src
							elif self.graph.edge_info[result].tgt not in self.graph.node_obj_lookup:
								raise ZenException, 'Missing endpoint: Target node (idx=%d) does not have an object' % self.graph.edge_info[result].tgt

							return self.graph.node_obj_lookup[self.graph.edge_info[result].src], self.graph.node_obj_lookup[self.graph.edge_info[result].tgt]
				except StopIteration:
					self.edge_iter = None
					curr_nidx = self.nbunch_iter.next()
					self.edge_iter = NodeEdgeIterator(self.graph,curr_nidx,self.which_degree,self.weight,self.data)

	def __iter__(SomeEdgeIterator self):
		return self

cdef class NeighborIterator:
	cdef NodeEdgeIterator inner_iter
	cdef bint data
	cdef deg
	cdef int nidx
	cdef DiGraph G
	cdef touched_nodes
	cdef bint use_nobjs
	cdef bint obj
	cdef long init_num_changes

	def __cinit__(NeighborIterator self, DiGraph G, int nidx,which_degree,obj,data,use_nobjs):
		self.init_num_changes = G.num_changes
		self.inner_iter = NodeEdgeIterator(G,nidx,which_degree,False)
		self.data = data
		self.deg = which_degree
		self.nidx = nidx
		self.G = G
		self.touched_nodes = set()
		self.use_nobjs = use_nobjs
		self.obj = obj

	def __next__(NeighborIterator self):
		if self.init_num_changes != self.G.num_changes:
			raise GraphChangedException()
		while True:
			eid = self.inner_iter.next()
			if not self.obj and not self.data:
				if self.nidx == self.G.edge_info[eid].src:
					if self.G.edge_info[eid].tgt in self.touched_nodes:
						continue
					self.touched_nodes.add(self.G.edge_info[eid].tgt)

					if self.use_nobjs:
						if self.G.edge_info[eid].tgt not in self.G.node_obj_lookup:
							raise ZenException, 'Node (idx=%d) does not have a node object' % self.G.edge_info[eid].tgt
						return self.G.node_obj_lookup[self.G.edge_info[eid].tgt]
					else:
						return self.G.edge_info[eid].tgt
				else:
					if self.G.edge_info[eid].src in self.touched_nodes:
						continue
					self.touched_nodes.add(self.G.edge_info[eid].src)

					if self.use_nobjs:
						if self.G.edge_info[eid].src not in self.G.node_obj_lookup:
							raise ZenException, 'Node (idx=%d) does not have a node object' % self.G.edge_info[eid].src
						return self.G.node_obj_lookup[self.G.edge_info[eid].src]
					else:
						return self.G.edge_info[eid].src
			elif self.obj and not self.data:
				val = None
				if self.nidx == self.G.edge_info[eid].src:
					if self.G.edge_info[eid].tgt in self.touched_nodes:
						continue
					self.touched_nodes.add(self.G.edge_info[eid].tgt)

					if self.use_nobjs:
						if self.G.edge_info[eid].tgt not in self.G.node_obj_lookup:
							raise ZenException, 'Node (idx=%d) does not have a node object' % self.G.edge_info[eid].tgt
						return self.G.node_obj_lookup[self.G.edge_info[eid].tgt], self.G.node_object(self.G.edge_info[eid].tgt)
					else:
						return self.G.edge_info[eid].tgt, self.G.node_object(self.G.edge_info[eid].tgt)
				else:
					if self.G.edge_info[eid].src in self.touched_nodes:
						continue
					self.touched_nodes.add(self.G.edge_info[eid].src)

					if self.use_nobjs:
						if self.G.edge_info[eid].src not in self.G.node_obj_lookup:
							raise ZenException, 'Node (idx=%d) does not have a node object' % self.G.edge_info[eid].src
						return self.G.node_obj_lookup[self.G.edge_info[eid].src], self.G.node_object(self.G.edge_info[eid].src)
					else:
						return self.G.edge_info[eid].src, self.G.node_object(self.G.edge_info[eid].src)
			elif not self.obj and self.data:
				val = None
				if self.nidx == self.G.edge_info[eid].src:
					if self.G.edge_info[eid].tgt in self.touched_nodes:
						continue
					self.touched_nodes.add(self.G.edge_info[eid].tgt)

					if self.use_nobjs:
						if self.G.edge_info[eid].tgt not in self.G.node_obj_lookup:
							raise ZenException, 'Node (idx=%d) does not have a node object' % self.G.edge_info[eid].tgt
						return self.G.node_obj_lookup[self.G.edge_info[eid].tgt], self.G.node_data_(self.G.edge_info[eid].tgt)
					else:
						return self.G.edge_info[eid].tgt, self.G.node_data_(self.G.edge_info[eid].tgt)
				else:
					if self.G.edge_info[eid].src in self.touched_nodes:
						continue
					self.touched_nodes.add(self.G.edge_info[eid].src)

					if self.use_nobjs:
						if self.G.edge_info[eid].src not in self.G.node_obj_lookup:
							raise ZenException, 'Node (idx=%d) does not have a node object' % self.G.edge_info[eid].src
						return self.G.node_obj_lookup[self.G.edge_info[eid].src], self.G.node_data_(self.G.edge_info[eid].src)
					else:
						return self.G.edge_info[eid].src, self.G.node_data_(self.G.edge_info[eid].src)
			else:
				if self.nidx == self.G.edge_info[eid].src:
					if self.G.edge_info[eid].tgt in self.touched_nodes:
						continue
					self.touched_nodes.add(self.G.edge_info[eid].tgt)

					if self.use_nobjs:
						if self.G.edge_info[eid].tgt not in self.G.node_obj_lookup:
							raise ZenException, 'Node (idx=%d) does not have a node object' % self.G.edge_info[eid].tgt
						return self.G.node_obj_lookup[self.G.edge_info[eid].tgt], self.G.node_object(self.G.edge_info[eid].tgt), self.G.node_data_(self.G.edge_info[eid].tgt)
					else:
						return self.G.edge_info[eid].tgt, self.G.node_object(self.G.edge_info[eid].tgt), self.G.node_data_(self.G.edge_info[eid].tgt)
				else:
					if self.G.edge_info[eid].src in self.touched_nodes:
						continue
					self.touched_nodes.add(self.G.edge_info[eid].src)

					if self.use_nobjs:
						if self.G.edge_info[eid].src not in self.G.node_obj_lookup:
							raise ZenException, 'Node (idx=%d) does not have a node object' % self.G.edge_info[eid].src
						return self.G.node_obj_lookup[self.G.edge_info[eid].src], self.G.node_object(self.G.edge_info[eid].src), self.G.node_data_(self.G.edge_info[eid].src)
					else:
						return self.G.edge_info[eid].src, self.G.node_object(self.G.edge_info[eid].src), self.G.node_data_(self.G.edge_info[eid].src)

	def __iter__(NeighborIterator self):
		return self

cdef class SomeNeighborIterator:
	cdef bint data
	cdef DiGraph graph
	cdef int idx
	cdef int which_degree
	cdef touched_nodes
	cdef nbunch_iter
	cdef neighbor_iter
	cdef bint use_nobjs
	cdef bint obj
	cdef long init_num_changes

	def __cinit__(SomeNeighborIterator self,DiGraph graph,nbunch,which_degree,obj,data,use_nobjs):
		self.init_num_changes = graph.num_changes
		self.graph = graph
		self.nbunch_iter = iter(nbunch)
		self.obj = obj
		self.data = data
		self.idx = 0
		self.which_degree = which_degree
		self.neighbor_iter = None
		self.touched_nodes = set()
		self.use_nobjs = use_nobjs

		# setup the first iterator
		if len(nbunch) > 0:
			curr_nidx = self.nbunch_iter.next()
			self.neighbor_iter = NeighborIterator(self.graph,curr_nidx,self.which_degree,self.obj,self.data,self.use_nobjs)
		else:
			self.neighbor_iter = None

	def __next__(SomeNeighborIterator self):
		if self.init_num_changes != self.graph.num_changes:
			raise GraphChangedException()

		while True:
			if self.neighbor_iter is None:
				raise StopIteration
			else:
				try:
					result = self.neighbor_iter.next()
					if self.data:
						if result[0] in self.touched_nodes:
							continue
						self.touched_nodes.add(result[0])
					else:
						if result in self.touched_nodes:
							continue
						self.touched_nodes.add(result)

					return result
				except StopIteration:
					self.neighbor_iter = None
					curr_nidx = self.nbunch_iter.next()
					self.neighbor_iter = NeighborIterator(self.graph,curr_nidx,self.which_degree,self.obj,self.data,self.use_nobjs)

	def __iter__(SomeNeighborIterator self):
		return self
