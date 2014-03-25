import numpy
import numpy as np
cimport numpy as np

import ctypes
from exceptions import *

cimport libc.stdlib as stdlib

# some things not defined in the cython stdlib header
cdef extern from "stdlib.h" nogil:
	void* memmove(void* destination, void* source, size_t size)

cdef extern from "math.h":
	double ceil(double x)
	double floor(double x)
	double fmax(double x, double y)
	double fmin(double x, double y)

__all__ = ['Graph']

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

When a node info entry is part of the free node list, the degree points to the next
entry in the list and the capacity points to the previous entry in the list.
"""
cdef struct NodeInfo:
	bint exists
	
	int degree # The number of entries in the edge list that are in use
	int* elist
	int capacity  # The length of the edge list
	
cdef int sizeof_NodeInfo = sizeof(NodeInfo)

"""
When an edge info entry is part of the free edge list, u points to the next entry
in the list and v points to the previous entry.
"""
cdef struct EdgeInfo:
	bint exists
	int u # Node idx for the node with smaller index
	int v # Node idx for the node with larger index
	double weight
	
cdef int sizeof_EdgeInfo = sizeof(EdgeInfo)

"""
The Graph class supports pickling.  Here we keep a record of changes from one pickle version to another.

Version 1.0:
	- The initial version (no diff)
"""
CURRENT_PICKLE_VERSION = 1.0

# static cython methods
cpdef Graph Graph_from_adj_matrix_np_ndarray(np.ndarray[np.float_t, ndim=2] M,
											node_obj_fxn):  
	cdef Graph G = Graph()
	cdef int i
	cdef int rows = M.shape[0]
	cdef int cols = M.shape[1]

	# add nodes
	G.add_nodes(rows,node_obj_fxn)

	# add edges
	for i in range(rows):
		for j in range(i,cols):
			if M[i,j] != 0:
				G.add_edge_(i,j,None,M[i,j])

	return G


cdef class Graph:
	"""
	This class provides a highly-optimized implementation of an `undirected graph <http://en.wikipedia.org/wiki/Undirected_graph#Undirected_graph>`_. 
	Duplicate edges are not allowed.
	
	Public properties include:
	
		* ``max_node_index`` (int): the largest node index currently in use
		* ``max_edge_index`` (int): the largest edge index currently in use
		* ``edge_list_capacity`` (int): the initial number of edge positions that will be allocated in a newly created node's edge list.
		* ``node_grow_factor`` (int): the multiple by which the node storage array will grow when its capacity is exceeded.
		* ``edge_grow_factor`` (int): the multiple by which the edge storage array will grow when its capacity is exceeded.
		* ``edge_list_grow_factor`` (int): the multiple by which the a node's edge list storage array will grow when its capacity is exceeded.
		
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
		Create a new :py:class:`Graph` from adjacency matrix information
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
			return Graph_from_adj_matrix_np_ndarray(<np.ndarray[np.float_t,ndim=2]> M,node_obj_fxn)
		else:
			raise TypeError, 'Objects of type %s cannot be handled as adjancency matrices' % type(M)


	def __init__(Graph self,**kwargs):
		"""
		Create a new :py:class:`Graph` object.
		
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
	
	def __dealloc__(Graph self):
		cdef int i
		# deallocate all node data (node_info)
		for i in range(self.next_node_idx):
			if self.node_info[i].exists:
				stdlib.free(self.node_info[i].elist)
		stdlib.free(self.node_info)
		
		# deallocate all edge data (edge_info)
		stdlib.free(self.edge_info)
	
	def __reduce__(self):
		return (Graph,tuple(),self.__getstate__())

	def __getstate__(self):
		
		state = {'PICKLE_VERSION':1.0}
		
		# store global details
		state['num_changes'] = self.num_changes
		
		# store all node details
		state['num_nodes'] = self.num_nodes
		state['node_capacity'] = self.node_capacity
		state['node_grow_factor'] = self.node_grow_factor		
		state['next_node_idx'] = self.next_node_idx
		state['max_node_idx'] = self.max_node_idx
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
								self.node_info[i].degree,
								self.node_info[i].capacity,
								[self.node_info[i].elist[j] for j in range(self.node_info[i].degree)] )
			else:
				pickle_entry = (bool(self.node_info[i].exists),
								self.node_info[i].degree,self.node_info[i].capacity)
				
			node_info.append(pickle_entry)
		
		state['node_info'] = node_info
		
		# store all edge details
		state['num_edges'] = self.num_edges
		state['edge_capacity'] = self.edge_capacity
		state['edge_grow_factor'] = self.edge_grow_factor		
		state['next_edge_idx'] = self.next_edge_idx
		state['max_edge_idx'] = self.max_edge_idx
		state['first_free_edge'] = self.first_free_edge		
		state['edge_data_lookup'] = self.edge_data_lookup
		
		# store edge_info
		edge_info = []
		for i in range(self.edge_capacity):
			edge_info.append( (	bool(self.edge_info[i].exists),
								self.edge_info[i].u,
								self.edge_info[i].v,
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
		self.max_node_idx = state['max_node_idx']
		self.first_free_node = state['first_free_node']
		self.node_obj_lookup = state['node_obj_lookup']
		self.node_data_lookup = state['node_data_lookup']
		self.node_idx_lookup = state['node_idx_lookup']
		self.edge_list_capacity = state['edge_list_capacity']
		self.edge_list_grow_factor = state['edge_list_grow_factor']
		
		# restore node_info
		self.node_info = <NodeInfo*> stdlib.malloc(sizeof_NodeInfo*self.node_capacity)
		for i,entry in enumerate(state['node_info']):
			
			if entry[0] is True:
				exists, degree, capacity, elist = entry
				self.node_info[i].exists = exists
				self.node_info[i].degree = degree
				self.node_info[i].capacity = capacity

				self.node_info[i].elist = <int*> stdlib.malloc(sizeof(int) * capacity)
				for j,eidx in enumerate(elist):
					self.node_info[i].elist[j] = eidx
						
			else:
				exists, degree, capacity = entry
				self.node_info[i].exists = exists
				self.node_info[i].degree = degree
				self.node_info[i].capacity = capacity
				
		# restore all edge details
		self.num_edges = state['num_edges']
		self.edge_capacity = state['edge_capacity']
		self.edge_grow_factor = state['edge_grow_factor']
		self.next_edge_idx = state['next_edge_idx']
		self.max_edge_idx = state['max_edge_idx']
		self.first_free_edge = state['first_free_edge']
		self.edge_data_lookup = state['edge_data_lookup']

		# restore edge_info
		self.edge_info = <EdgeInfo*> stdlib.malloc(sizeof_EdgeInfo*self.edge_capacity)
		for i,entry in enumerate(state['edge_info']):
			exists,u,v,weight = entry
			self.edge_info[i].exists = exists
			self.edge_info[i].u = u
			self.edge_info[i].v = v
			self.edge_info[i].weight = weight
				
		return
	
	# def __getattr__(self,name):
	# 	raise AttributeError, 'Class has no attribute "%s"' % name
	
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
			
			assert self.node_info[i].capacity == j, 'Free node %d points to the incorrect predecessor (%d correct, %d actual)' % (i,j,self.node_info[i].capacity)
			
			j = i
			i = self.node_info[i].degree
			
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
			
			assert self.edge_info[i].v == j, 'Free edge %d points to the incorrect predecessor (%d correct, %d actual)' % (i,j,self.edge_info[i].v)
			
			j = i
			i = self.edge_info[i].u
			
		assert (num_free_edges + num_existing_edges) == self.next_edge_idx, '(# free edges) + (# existing edges) != self.next_edge_idx (%d + %d != %d)' % (num_free_edges,num_existing_edges,self.next_edge_idx)
		
	cpdef copy(Graph self):
		"""
		Create a copy of this graph.  

		.. note:: that node and edge indices are preserved in this copy.

		**Returns**: 
			:py:class:`zen.Graph`. A new graph object that contains an independent copy of the connectivity of this graph.  
			Node objects and node/edge data in the new graph reference the same objects as in the old graph.
		"""
		cdef Graph G = Graph()
		self.__copy_graph_self_into(G)

		return G

	cdef __copy_graph_self_into(Graph self,Graph G):
		"""
		A helper method for copy functions of Graph and subclasses.  This method copies the graph itself into
		the graph provided.  This method doesn't return anything since the graph is modified in place.

		.. note:: per the contract of the :py:meth:`Graph.copy` method, the node and edge indices are preserved
		          in this copy.

		**Args**:

			* ``G`` (Graph): the graph that will be populated with all content.  Note that this graph should be empty.

		"""
		cdef int i,j,eidx,eidx2
		cdef double weight

		for i in range(self.next_node_idx):
			if self.node_info[i].exists:
				nobj = self.node_object(i)
				ndata = self.node_data_(i)

				# this will preserve node index
				G.add_node_x(i,G.edge_list_capacity,nobj,ndata)

		for eidx in range(self.next_edge_idx):
			if self.edge_info[eidx].exists:
				i = self.edge_info[eidx].u
				j = self.edge_info[eidx].v
				edata = self.edge_data_(eidx)
				weight = self.weight_(eidx)

				# this will preserve edge index
				G.add_edge_x(eidx,i,j,edata,weight)

		return G
	
	cpdef bint is_directed(Graph self):
		"""
		Return ``True`` if this graph is directed (which it is not).
		"""
		return False
	
	cpdef bint is_compact(Graph self):
		"""
		Return ``True`` if the graph is in compact form.  
		
		A graph is compact if there are no unallocated node or edge indices.
		The graph can be compacted by calling the :py:meth:`Graph.compact` method.
		"""
		return (self.num_nodes == (self.max_node_idx+1) and self.num_edges == (self.max_edge_idx+1))
	
	cpdef compact(Graph self):
		"""
		Compact the graph in place.  This will re-assign:
		
			#. node indices such that there are no unallocated node indices less than self.max_node_idx
			#. edge indices such that there are no unallocated edge indices less than self.max_edge_idx
			
		.. note:: At present no way is provided of keeping track of the changes made to node and edge indices.
		"""
		cdef int next_free_idx
		cdef int src, dest
		cdef int u,v,i,nidx
		cdef int eidx
		
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
			# modify all the edges and the relevant nodes
			for i in range(self.node_info[dest].degree):
				eidx = self.node_info[dest].elist[i]
				
				# get the other node whose edge list we need to update
				v = -1
				if self.edge_info[eidx].u == src:
					v = self.edge_info[eidx].v
				elif self.edge_info[eidx].v == src:
					v = self.edge_info[eidx].u
				else: 
					# there is a self-loop and this is one of the two entries.  Furthermore, it's already been processed
					# so there's nothing to do.
					continue
				
				#####
				# update the other node's edge list
				#print '\tv = %d' % v
				
				# remove the entry for src
				self.__remove_edge_from_edgelist(v,eidx,src)
				
				# if the edge is a self-loop, remove it twice
				if src == v:
					self.__remove_edge_from_edgelist(v,eidx,src)

					# update v to the new value (since it's a self-loop)
					v = dest
					
					# the source node degree was changed, but not the dest degree
					# we accomodate this here
					self.node_info[v].degree -= 2
					
				# update the edge endpoints
				if dest < v:
					self.edge_info[eidx].u = dest
					self.edge_info[eidx].v = v
				else:
					self.edge_info[eidx].u = v
					self.edge_info[eidx].v = dest				
				
				# insert the entry for dest
				self.__insert_edge_into_edgelist(v,eidx,dest)
				
				# if the edge is a self-loop, insert it twice
				if dest == v:
					self.__insert_edge_into_edgelist(v,eidx,dest)
			
			# update the max node index
			while self.max_node_idx >= 0 and not self.node_info[self.max_node_idx].exists:
				self.max_node_idx -= 1
								
			# move to the next node
			nidx += 1
		
		# at this point, node info has been defragmented so there are no free nodes
		# This means that the next unallocated node is the entry right after the max node idx.
		self.first_free_node = -1		
		self.next_node_idx = self.max_node_idx + 1
		
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
			u = self.edge_info[dest].u
			v = self.edge_info[dest].v

			# in u
			i = self.find_elist_insert_pos(self.node_info[u].elist,self.node_info[u].degree,u,v)
			self.node_info[u].elist[i] = dest

			# in v
			if u == v: # if this is a self-loop...
				# find_elist_insert_pos always finds the lower index (if there are two identical entries)
				# so we also reset the next one up
				self.node_info[v].elist[i+1] = dest 
			else:
				i = self.find_elist_insert_pos(self.node_info[v].elist,self.node_info[v].degree,v,u)
				self.node_info[v].elist[i] = dest
			
			# wipe out the source edge info
			self.edge_info[src].u = -1
			self.edge_info[src].v = -1
			
			# update the max node index
			while self.max_edge_idx >= 0 and not self.edge_info[self.max_edge_idx].exists:
				self.max_edge_idx -= 1
				
			# move to the next edge
			eidx += 1
			
		# at this point the max number of nodes and the next node available are adjacent
		self.first_free_edge = -1
		self.next_edge_idx = self.max_edge_idx + 1
		
		return
	
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
				
					G = Graph()
					G.add_node('a') # this node has index 0
					G.add_node('b') # this node has index 1
					G.add_node('c') # this node has index 2
					
					G.rm_node('b') # after this point, index 1 doesn't correspond to a valid node
					
					M = G.matrix() # M is a 3x3 matrix
					
					V = M[1,:] # the values in V are meaningless because a node with index 1 doesn't exist
					
				This situation can be resolved by making a call to :py:meth:`Graph.compact` prior to calling this function::
				
					G = Graph()
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
				for j in range(self.node_info[u].degree):
					eidx = self.node_info[u].elist[j]
					v = self.endpoint_(eidx,u)
					w = self.edge_info[eidx].weight
					A[u,v] = w
					A[v,u] = w
					
		return A
		
	cpdef np.ndarray[np.int_t] add_nodes(Graph self,int num_nodes,node_obj_fxn=None):
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
		cdef np.ndarray[np.int_t,ndim=1] indexes = np.empty(num_nodes, np.int)
		cdef int node_idx
		
		self.num_changes += 1
		
		for nn_count in range(num_nodes):
			
			# recycle an unused node index if possible
			if self.first_free_node != -1:
				node_idx = self.first_free_node
			else:
				node_idx = self.next_node_idx

			indexes[nn_count] = node_idx
			
			nobj = None
			if node_obj_fxn is not None:
				nobj = node_obj_fxn(node_idx)
				
			self.add_node_x(node_idx,self.edge_list_capacity,nobj,None)
			
		return indexes
		
	cpdef int add_node(Graph self,nobj=None,data=None) except -1:
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
		cdef int next_free_node
		
		# recycle an unused node index if possible
		if self.first_free_node != -1:
			node_idx = self.first_free_node
		else:
			node_idx = self.next_node_idx
			
		self.add_node_x(node_idx,self.edge_list_capacity,nobj,data)
		
		return node_idx
	
	cdef add_to_free_node_list(self,int nidx):
		self.node_info[nidx].degree = self.first_free_node
		self.node_info[nidx].capacity = -1
		
		if self.first_free_node != -1:
			self.node_info[self.first_free_node].capacity = nidx
			
		self.first_free_node = nidx
			
	cdef remove_from_free_node_list(self,int nidx):
		cdef int prev_free_node = self.node_info[nidx].capacity
		cdef int next_free_node = self.node_info[nidx].degree
		
		if prev_free_node == -1:
			self.first_free_node = next_free_node
		else:
			self.node_info[prev_free_node].degree = next_free_node
			
		if next_free_node != -1:
			self.node_info[next_free_node].capacity = prev_free_node
			
	cpdef add_node_x(Graph self,int node_idx,int edge_list_capacity,nobj,data):
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
			* ``edge_list_capacity`` (int): the number of entries that should be allocated in the edge list for this node.
			* ``nobj``: the node object that will be associated with this node.  If ``None``, then no object will be
				assigned to this node.
			* ``data``: the data object that will be associated with this node.  If ``None``, then no data will be 
				assigned to this node.
				
		**Raises**:
			:py:exc:`ZenException`: if the node index is already in use or the node object is not unique.
		"""
		cdef int i
		
		if node_idx < self.node_capacity and self.node_info[node_idx].exists == True:
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
		if node_idx >= self.node_capacity:
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
		self.node_info[node_idx].degree = 0
		
		# initialize edge lists
		self.node_info[node_idx].elist = <int*> stdlib.malloc(sizeof(int) * self.edge_list_capacity)
		self.node_info[node_idx].capacity = self.edge_list_capacity
		
		self.num_nodes += 1
		
		# notify listeners if necessary
		if self.num_graph_listeners > 0:
			for listener in self.graph_listeners:
				listener.node_added(node_idx,nobj,data)
		
		return
		
	def __contains__(Graph self,nobj):
		"""
		Return ``True`` if object ``nobj`` is associated with a node in this graph.
		"""
		return nobj in self.node_idx_lookup
	
	cpdef int node_idx(Graph self,nobj) except -1:
		"""
		Return the index of the node with node object ``nobj``.
		"""
		return self.node_idx_lookup[nobj]
		
	cpdef node_object(Graph self,int nidx):
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
	
	cpdef set_node_data(Graph self,nobj,data):
		"""
		Associate a new data object with a specific node in the network.
		If data is None, then any data associated with the node is deleted.
		
		**Args**:
		
			* ``nobj``: the node object identifying the node whose data association is being changed.
			* ``data``: the data object to associate.  If ``None``, then any data object currently
				associated with this node will be deleted.
		"""
		self.set_node_data_(self.node_idx_lookup[nobj],data)

	cpdef set_node_data_(Graph self,int nidx,data):
		"""
		Associate a new data object with a specific node in the network.
		If data is None, then any data associated with the node is deleted.
	
		**Args**:
	
			* ``nidx``: the index of the node whose data association is being changed.
			* ``data``: the data object to associate.  If ``None``, then any data object currently
				associated with this node will be deleted.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx
			
		if data == None:
			if nidx in self.node_data_lookup:
				del self.node_data_lookup[nidx]
		else:
			self.node_data_lookup[nidx] = data
						
	cpdef node_data(Graph self,nobj):
		"""
		Return the data object associated with node having object identifier ``nobj``.
		"""
		return self.node_data_(self.node_idx_lookup[nobj])

	cpdef node_data_(Graph self,int nidx):
		"""
		Return the data object associated with node having index ``nidx``.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx
			
		if nidx in self.node_data_lookup:
			return self.node_data_lookup[nidx]
		else:
			return None
		
	cpdef nodes_iter(Graph self,data=False):
		"""
		Return an iterator over all the nodes in the graph.  
		
		By default, the iterator yields node objects.  If ``data`` is ``True``,
		then the iterator yields tuples ``(node_object,node_data)``.
		"""
		return NodeIterator(self,False,data,True)

	cpdef nodes_iter_(Graph self,obj=False,data=False):
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
				
	cpdef nodes(Graph self,data=False):
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

	cpdef nodes_(Graph self,obj=False,data=False):
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
		
	cpdef rm_node(Graph self,nobj):
		"""
		Remove the node associated with node object ``nobj``.  Any edges incident to the node are also removed
		from the graph.
		"""
		self.rm_node_(self.node_idx_lookup[nobj])
	
	cpdef rm_node_(Graph self,int nidx):
		"""
		Remove the node with index ``nidx``. Any edges incident to the node are also removed from the graph.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx
			
		cdef int i
		
		self.num_changes += 1
		
		# disconnect all inbound edges
		cdef last_val = -1
		for i in range(self.node_info[nidx].degree-1,-1,-1):
			eidx = self.node_info[nidx].elist[i]
			if eidx == last_val:
				continue
			last_val = eidx
			
			# remove the edge
			self.rm_edge_(eidx)
		
		# free up all the data structures
		self.node_info[nidx].exists = False
		stdlib.free(self.node_info[nidx].elist)
	
		if nidx in self.node_data_lookup:
			del self.node_data_lookup[nidx]
			
		nobj = None	
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
		
		# update the node count						
		self.num_nodes -= 1
		
		# notify listeners if necessary
		if self.num_graph_listeners > 0:
			for listener in self.graph_listeners:
				listener.node_removed(nidx,nobj)
				
	
	cpdef degree(Graph self,nobj):
		"""
		Return the degree of node with object ``nobj``.
		"""
		return self.degree_(self.node_idx_lookup[nobj])

	cpdef degree_(Graph self,int nidx):
		"""
		Return the degree of node with index ``nidx``.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx
			
		return self.node_info[nidx].degree
	
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
	
	cdef add_to_free_edge_list(self,int eidx):
		self.edge_info[eidx].u = self.first_free_edge
		self.edge_info[eidx].v = -1

		if self.first_free_edge != -1:
			self.edge_info[self.first_free_edge].v = eidx

		self.first_free_edge = eidx

	cdef remove_from_free_edge_list(self,int eidx):
		cdef int prev_free_edge = self.edge_info[eidx].v
		cdef int next_free_edge = self.edge_info[eidx].u

		if prev_free_edge == -1:
			self.first_free_edge = next_free_edge
		else:
			self.edge_info[prev_free_edge].u = next_free_edge

		if next_free_edge != -1:
			self.edge_info[next_free_edge].v = prev_free_edge
	
	cpdef int add_edge(self, u, v, data=None, double weight=1) except -1:
		"""
		Add an edge to the graph.
		
		As a convenience, if ``u`` or ``v`` are not valid node objects, they will be added to the graph
		and then the edge will be added.::
		
			G = Graph()
			
			print len(G) # prints '0'
			G.add_edge(1,2) # First nodes 1 and 2 will be added, then the edge will be added
			print len(G) # prints '2' since there are now two nodes in the graph.
			
		.. note::
			In undirected graphs, the edges (u,v) and (v,u) refer to the same edge.  Thus the following code
			will raise an error::
			
				G = Graph()
				G.add_edge(1,2)
				G.add_edge(2,1) # a ZenException will be raised because the edge already exists.
			
		
		**Args**:
		
			* ``u``: one endpoint of the graph. 
			* ``v``: another endpoint of the graph.
			* ``data [=None]``: an optional data object to associate with the edge
			* ``weight [=1]`` (float): the weight of the edge.
			
		**Returns**:
			``integer``. The index for the newly created edge.
			
		**Raises**:
			:py:exc:`ZenException`: if the edge already exists in the graph.
			
		"""
		cdef int nidx1, nidx2
		
		if u not in self.node_idx_lookup:
			nidx1 = self.add_node(u)
		else:
			nidx1 = self.node_idx_lookup[u]
		
		if v not in self.node_idx_lookup:
			nidx2 = self.add_node(v)
		else:
			nidx2 = self.node_idx_lookup[v]
		
		return self.add_edge_(nidx1,nidx2,data,weight)
	
	cpdef int add_edge_(Graph self, int u, int v, data=None, double weight=1) except -1:
		"""
		Add an edge to the graph.
		
		This version of the edge addition functionality uses node indices (not node objects).
		Unlike in :py:method:``.add_edge``, if ``u`` or ``v`` are not valid node indices, then an
		exception will be raised.
			
		.. note::
			In undirected graphs, the edges (u,v) and (v,u) refer to the same edge.  Thus the following code
			will raise an error::
			
				G = Graph()
				n1 = G.add_node(1)
				n2 = G.add_node(2)
				G.add_edge_(n1,n2)
				G.add_edge(n2,n1) # a ZenException will be raised because the edge already exists.
			
		
		**Args**:
		
			* ``u`` (int): one endpoint of the graph. This is a node index.
			* ``v`` (int): another endpoint of the graph. This is a node index.
			* ``data [=None]``: an optional data object to associate with the edge
			* ``weight [=1]`` (float): the weight of the edge.
			
		**Returns**:
			``integer``. The index for the newly created edge.
			
		**Raises**:
			:py:exc:`ZenException`: if the edge already exists in the graph or if either of the node indices are invalid.
		"""	
		cdef int tmp
		
		self.num_changes += 1
		
		# order u and v
		if u > v:
			tmp = u
			u = v
			v = tmp
		
		if u >= self.node_capacity or not self.node_info[u].exists or v >= self.node_capacity or not self.node_info[v].exists:
			raise ZenException, 'Both source and destination nodes must exist (%d,%d)' % (u,v)
			
		cdef int eidx
		if self.first_free_edge != -1:
			eidx = self.first_free_edge
		else:
			eidx = self.next_edge_idx
	
		self.add_edge_x(eidx,u,v,data,weight)

		return eidx

	cpdef add_edge_x(self, int eidx, int u, int v, data, double weight):
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
			* ``u`` (int): one endpoint of the edge. This is a node index.
			* ``v`` (int): the other endpoint of the edge. This is a node index.
			* ``nobj``: the node object that will be associated with this node.  If ``None``, then no object will be
				assigned to this node.
			* ``data``: the data object that will be associated with this node.  If ``None``, then no data will be 
				assigned to this node.
			
		**Raises**:
			:py:exc:`ZenException`: if the edge already exists in the graph, the edge index is already in use, or either of the
			node indices are invalid.
		"""
		cdef int i
	
		if eidx < self.edge_capacity and self.edge_info[eidx].exists == True:
			raise ZenException, 'Adding edge at index %d will overwrite an existing edge' % eidx
		
		# grow the info array
		cdef int new_edge_capacity
		if eidx >= self.edge_capacity:
			new_edge_capacity = <int> ceil( <float>self.edge_capacity * self.edge_grow_factor)
			self.edge_info = <EdgeInfo*> stdlib.realloc( self.edge_info, sizeof_EdgeInfo * new_edge_capacity)
			for i in range(self.edge_capacity,new_edge_capacity):
				self.edge_info[i].exists = False
				
			# add all the newly allocated empty edges to the free edge list
			for i in range(self.edge_capacity,eidx):
				self.add_to_free_edge_list(i)
					
			self.edge_capacity = new_edge_capacity
		
		####
		# connect up the edges to nodes
		
		# Note: this is where duplicate edges are detected.  So
		# an exception can be thrown by these two functions.  Hence,
		# it's necessary to make modifications to the edge list *AFTER*
		# these functions both successfully return.
		
		# u
		self.__insert_edge_into_edgelist(u,eidx,v)
		
		# v
		self.__insert_edge_into_edgelist(v,eidx,u)
		
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
			
		if data is not None:
			self.edge_data_lookup[eidx] = data
			
		### Add edge info
		self.edge_info[eidx].exists = True
		self.edge_info[eidx].u = u
		self.edge_info[eidx].v = v
		self.edge_info[eidx].weight = weight
		
		#####
		# Done
		self.num_edges += 1
		
		# notify listeners if necessary
		if self.num_graph_listeners > 0:
			for listener in self.graph_listeners:
				listener.edge_added(eidx,u,v,data,weight)
		
		return
	
	cdef __insert_edge_into_edgelist(Graph self, int u, int eidx, int v):
		"""
		Insert edge eidx with other endpoint v into the edgelist of node u.
		"""
		cdef int new_capacity
		cdef int num_edges = self.node_info[u].degree
		cdef int pos = 0
		cdef double elist_len = <double> self.node_info[u].capacity
		if num_edges >= elist_len:
			new_capacity = <int>ceil(elist_len * self.edge_list_grow_factor)
			self.node_info[u].elist = <int*> stdlib.realloc(self.node_info[u].elist, sizeof(int) * new_capacity)
			self.node_info[u].capacity = new_capacity
		cdef int* elist = self.node_info[u].elist
		pos = self.find_elist_insert_pos(elist,num_edges,u,v)
		if pos == num_edges:
			elist[pos] = eidx
		elif u < v and self.edge_info[elist[pos]].v == v:
			raise ZenException, 'Duplicate edges (%d,%d) are not permitted in a Graph' % (u,v)
		else:
			memmove(elist + (pos+1),elist + pos,(num_edges-pos) * sizeof(int))
			elist[pos] = eidx
		self.node_info[u].degree += 1
		
		return
	
	cdef int __endpoint(Graph self, EdgeInfo ei, int this_nidx):
		if ei.u == this_nidx:
			return ei.v
		else:
			return ei.u
	
	cdef int find_elist_insert_pos(Graph self, int* elist, int elist_len, int this_nidx, int nidx):
		"""
		Perform a binary search for the insert position
		"""
		cdef int pos = <int> floor(elist_len/2)
		
		if elist_len == 0:
			return 0
			
		if pos == 0:
			if self.__endpoint(self.edge_info[elist[pos]],this_nidx) < nidx:
				return elist_len
			else:
				return 0
		
		while True:
			if pos == 0:
				return 0
			elif pos == (elist_len-1) and self.__endpoint(self.edge_info[elist[pos]],this_nidx) < nidx:
				return elist_len
			elif (self.__endpoint(self.edge_info[elist[pos-1]],this_nidx) < nidx and self.__endpoint(self.edge_info[elist[pos]],this_nidx) >= nidx):
				return pos
			else:
				if self.__endpoint(self.edge_info[elist[pos]],this_nidx) < nidx:
					pos = pos + <int> floor((elist_len-pos)/2)
				else:
					elist_len = pos
					pos = <int> floor(pos/2)
	
	cpdef rm_edge(Graph self,u,v):
		"""
		Remove the edge between node objects ``u`` and ``v``.
		
		**Raises**:
			
			* :py:exc:`ZenException`: if the edge index is invalid.
			* :py:exc:`KeyError`: if the node objects are invalid.
		"""
		self.rm_edge_(self.edge_idx(u,v))
	
	cpdef rm_edge_(Graph self,int eidx):
		"""
		Remove the edge with index ``eidx``.
		
		**Raises**:
			:py:exc:`ZenException`: if ``eid`` is an invalid edge index.
		"""
		if eidx >= self.edge_capacity:
			raise ZenException, 'Invalid edge idx %d' % eidx
		elif not self.edge_info[eidx].exists:
			raise ZenException, 'Edge idx %d refers to a non-existant edge' % eidx
			
		cdef int i
		
		self.num_changes += 1
		
		#####
		# remove entries in u and v
		cdef int u = self.edge_info[eidx].u
		cdef int v = self.edge_info[eidx].v
		
		# in u
		self.__remove_edge_from_edgelist(u,eidx,v)

		# in v
		self.__remove_edge_from_edgelist(v,eidx,u)
		
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
				listener.edge_removed(eidx,u,v)
				
		return
	
	cdef __remove_edge_from_edgelist(Graph self, int u, int eidx, int v):
		cdef int i = self.find_elist_insert_pos(self.node_info[u].elist,self.node_info[u].degree,u,v)
		memmove(&self.node_info[u].elist[i],&self.node_info[u].elist[i+1],(self.node_info[u].degree-i-1)*sizeof(int))
		self.node_info[u].degree -= 1
		
		return
	
	cpdef endpoints(Graph self,int eidx):
		"""
		Return the node objects at the endpoints of the edge with index ``eidx``.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Edge index %d does not exist' % eidx
	
		return self.node_obj_lookup[self.edge_info[eidx].u], self.node_obj_lookup[self.edge_info[eidx].v]
		
	cpdef endpoints_(Graph self,int eidx):
		"""
		Return the node indices at the endpoints of the edge with index ``eidx``.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx

		return self.edge_info[eidx].u, self.edge_info[eidx].v
	
	cpdef endpoint(Graph self,int eidx,u):
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
	
	cpdef int endpoint_(Graph self,int eidx,int u) except -1:
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
			
		if u == self.edge_info[eidx].u:
			return self.edge_info[eidx].v
		else:
			return self.edge_info[eidx].u
		
	cpdef set_weight(Graph self,u,v,double w):
		"""
		Set the weight of the edge between nodes ``u`` and ``v`` (node objects) to ``w``.
		"""
		self.set_weight_(self.edge_idx(u,v),w)

	cpdef set_weight_(Graph self,int eidx,double w):
		"""
		Set the weight of the edge with index ``eidx`` to ``w``.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx
		
		self.edge_info[eidx].weight = w
		
	cpdef double weight(Graph self,u,v):
		"""
		Return the weight of the edge between nodes ``u`` and ``v`` (node objects).
		"""
		return self.weight_(self.edge_idx(u,v))

	cpdef double weight_(Graph self,int eidx):
		"""
		Return the weight of the edge with index ``eidx``.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx
			
		return self.edge_info[eidx].weight
		
	cpdef set_edge_data(Graph self,u,v,data):
		"""
		Associate a data object with the edge between nodes ``u`` and ``v`` (node objects).
		
		The value of ``data`` will replace any data object currently associated with the edge.
		If data is None, then any data associated with the edge is deleted.
		"""
		self.set_edge_data_(self.edge_idx(u,v),data)
		
	cpdef set_edge_data_(Graph self,int eidx,data):
		"""
		Associate a data object with the edge with index ``eidx``.
	
		The value of ``data`` will replace any data object currently associated with the edge.
		If data is None, then any data associated with the edge is deleted.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx
			
		if data == None:
			if eidx in self.edge_data_lookup:
				del self.edge_data_lookup[eidx]
		else:
			self.edge_data_lookup[eidx] = data

	cpdef edge_data(Graph self,u,v):
		"""
		Return the data associated with the edge between ``u`` and ``v`` (node objects).
		"""
		return self.edge_data_(self.edge_idx(u,v))
		
	cpdef edge_data_(Graph self,int eidx):
		"""
		Return the data associated with the edge with index ``eidx``.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx
	
		if eidx in self.edge_data_lookup:
			return self.edge_data_lookup[eidx]
		else:
			return None
			
	cpdef bint has_edge(Graph self,u,v):
		"""
		Return ``True`` if the graph contains an edge between ``u`` and ``v`` (node objects).  
		If either node object is not in the graph, this method returns ``False``.
		"""
		if u not in self.node_idx_lookup:
			return False
		if v not in self.node_idx_lookup:
			return False
			
		u = self.node_idx_lookup[u]
		v = self.node_idx_lookup[v]
		
		return self.has_edge_(u,v)
		
	cpdef bint has_edge_(Graph self,int u,int v):
		"""
		Return ``True`` if the graph contains an edge between ``u`` and ``v`` (node indices).
		
		**Raises**:
			:py:exc:`ZenException`: if either ``u`` or ``v`` are invalid node indices.
		"""
		if u >= self.node_capacity or not self.node_info[u].exists:
			raise ZenException, 'Invalid node (u) index %d' % u
			
		if v >= self.node_capacity or not self.node_info[v].exists:
			raise ZenException, 'Invalid node (v) index %d' % v
			
		cdef int tmp
		if u > v:
			tmp = u
			u = v
			v = tmp
		
		cdef int num_edges = self.node_info[u].degree
		cdef int* elist = self.node_info[u].elist
		cdef int pos = self.find_elist_insert_pos(elist,self.node_info[u].degree,u,v)
		return pos < self.node_info[u].degree and self.edge_info[elist[pos]].v == v
	
	cpdef edge_idx(Graph self, u, v):
		"""
		Return the edge index for the edge between ``u`` and ``v`` (node objects).
		"""
		u = self.node_idx_lookup[u]
		v = self.node_idx_lookup[v]
		return self.edge_idx_(u,v) #,data)
	
	cpdef edge_idx_(Graph self, int u, int v):
		"""
		Return the edge index for the edge between ``u`` and ``v`` (node indices).
		"""
		if u >= self.node_capacity or not self.node_info[u].exists:
			raise ZenException, 'Invalid node (u) index %d' % u
			
		if v >= self.node_capacity or not self.node_info[v].exists:
			raise ZenException, 'Invalid node (v) index %d' % v
			
		cdef int tmp
		if u > v:
			tmp = u
			u = v
			v = tmp
			
		cdef int num_edges = self.node_info[u].degree
		cdef int* elist = self.node_info[u].elist
		cdef int pos = self.find_elist_insert_pos(elist,self.node_info[u].degree,u,v)
		
		if pos < self.node_info[u].degree and self.edge_info[elist[pos]].v == v:
			return elist[pos]
		else:			
			raise ZenException, 'Edge (%d,%d) does not exist.' % (u,v)
	
	cpdef edges_iter(Graph self,nobj=None,bint data=False,bint weight=False):
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
		
			G = Graph()
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
				print u,v,data
		"""
		if nobj is None:
			return AllEdgeIterator(self,weight,data,True)
		else:
			return NodeEdgeIterator(self,self.node_idx_lookup[nobj],weight,data,True)
	
	cpdef edges_iter_(Graph self,int nidx=-1,bint data=False,bint weight=False):
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
	
			G = Graph()
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
				
			return NodeEdgeIterator(self,nidx,weight,data,False)
	
	cpdef edges(Graph self,nobj=None,bint data=False,bint weight=False):
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
	
			G = Graph()
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
				print u,v,data
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
					if self.edge_info[i].u not in self.node_obj_lookup or self.edge_info[i].v not in self.node_obj_lookup:
						raise ZenException, 'Edge (idx=%d) does not have endpoints with node objects' % i
						
					if data is True:
						if weight is True:
							result.append( (self.node_obj_lookup[self.edge_info[i].u],self.node_obj_lookup[self.edge_info[i].v],self.edge_data_(i),self.edge_info[i].weight) )
						else:
							result.append( (self.node_obj_lookup[self.edge_info[i].u],self.node_obj_lookup[self.edge_info[i].v],self.edge_data_(i)) )
					else:
						if weight:
							result.append( (self.node_obj_lookup[self.edge_info[i].u],self.node_obj_lookup[self.edge_info[i].v], self.edge_info[i].weight) )
						else:
							result.append( (self.node_obj_lookup[self.edge_info[i].u],self.node_obj_lookup[self.edge_info[i].v]) )
					idx += 1
					
			return result
		else:
			idx = 0
			num_edges = self.node_info[nidx].degree
			elist = self.node_info[nidx].elist
			for i in range(num_edges):
				if self.edge_info[i].u not in self.node_obj_lookup or self.edge_info[i].v not in self.node_obj_lookup:
					raise ZenException, 'Edge (idx=%d) does not have endpoints with node objects' % i
					
				if data is True:
					if weight is True:
						result.append( (self.node_obj_lookup[self.edge_info[elist[i]].u],self.node_obj_lookup[self.edge_info[elist[i]].v],self.edge_data_(elist[i]), self.edge_info[elist[i]].weight) )
					else:
						result.append( (self.node_obj_lookup[self.edge_info[elist[i]].u],self.node_obj_lookup[self.edge_info[elist[i]].v],self.edge_data_(elist[i])) )
				else:
					if weight:
						result.append( (self.node_obj_lookup[self.edge_info[elist[i]].u],self.node_obj_lookup[self.edge_info[elist[i]].v],self.edge_info[elist[i]].weight) )
					else:
						result.append( (self.node_obj_lookup[self.edge_info[elist[i]].u],self.node_obj_lookup[self.edge_info[elist[i]].v]) )
				idx += 1
			
			return result
				
	cpdef edges_(Graph self,int nidx=-1,bint data=False,bint weight=False):
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

			G = Graph()
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
		cdef int num_edges
		cdef int* elist
		cdef int i
		
		# iterate over all edges
		result = None
		if nidx == -1:
			if data and weight:
				result = numpy.empty( (self.num_edges, 3), dtype=numpy.object_)
			if data or weight:
				result = numpy.empty( (self.num_edges, 2), dtype=numpy.object_)
			else:
				result = numpy.empty(self.num_edges, dtype=numpy.object_)
				
			idx = 0
			for i in range(self.next_edge_idx):
				if self.edge_info[i].exists:
					if data and weight:
						result[idx,0] = i
						result[idx,1] = self.edge_data_(i)
						result[idx,2] = self.edge_info[i].weight
					elif data is True:
						result[idx,0] = i
						result[idx,1] = self.edge_data_(i)
					elif weight:
						result[idx,0] = i
						result[idx,1] = self.edge_info[i].weight
					else:
						result[idx] = i
					idx += 1
					
			return result
		else:
			if nidx >= self.node_capacity or not self.node_info[nidx].exists:
				raise ZenException, 'Invalid node idx %d' % nidx
				
			if data:
				result = numpy.empty( (self.node_info[nidx].degree,2), dtype=numpy.object_)
			else:
				result = numpy.empty(self.node_info[nidx].degree, dtype=numpy.object_)
			
			idx = 0
			num_edges = self.node_info[nidx].degree
			elist = self.node_info[nidx].elist
			for i in range(num_edges):
				if data and weight:
					result[idx,0] = elist[i]
					result[idx,1] = self.edge_data_(elist[i])
					result[idx,2] = self.edge_info[elist[i]].weight					
				elif data is True:
					result[idx,0] = elist[i]
					result[idx,1] = self.edge_data_(elist[i])
				elif weight:
					result[idx,0] = elist[i]
					result[idx,1] = self.edge_info[elist[i]].weight					
				else:
					result[idx] = elist[i]
				idx += 1
			
			return result
	
	cpdef grp_edges_iter(Graph self,nbunch,bint data=False,bint weight=False):
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
		return SomeEdgeIterator(self,[self.node_idx_lookup[x] for x in nbunch],weight,data,True)
		
	cpdef grp_edges_iter_(Graph self,nbunch,bint data=False,bint weight=False):
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
				
		return SomeEdgeIterator(self,nbunch,weight,data)
						
	cpdef neighbors(Graph self,nobj,data=False):
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
		num_edges = self.node_info[nidx].degree
		elist = self.node_info[nidx].elist
		for i in range(num_edges):
			rid = self.edge_info[elist[i]].u
			if rid == nidx:
				rid = self.edge_info[elist[i]].v
			
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
		
	cpdef neighbors_(Graph self,int nidx,obj=False,data=False):
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
			result = numpy.empty( (self.node_info[nidx].degree,ndim), dtype=numpy.object_)
		else:
			result = numpy.empty( self.node_info[nidx].degree, dtype=numpy.object_)
		
		idx = 0
		
		# loop over in edges
		num_edges = self.node_info[nidx].degree
		elist = self.node_info[nidx].elist
		for i in range(num_edges):
			rid = self.edge_info[elist[i]].u
			if rid == nidx:
				rid = self.edge_info[elist[i]].v
			
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
		
	cpdef neighbors_iter(Graph self,nobj,data=False):
		"""
		Return an iterator over a node's immediate neighbors.

		By default, the iterator will yield the node object for each immediate neighbor of ``nobj``.

		**Args**:

			* ``nobj``: this is the node object identifying the node whose neighbors to iterate over.

			* ``data [=False]`` (boolean): if ``True``, then a tuple is returned (rather than a node object) 
				containing the node	object and the data object associated with the node (e.g., ``(n,d)``).
		"""
		return NeighborIterator(self,self.node_idx_lookup[nobj],False,data,True)
		
	cpdef neighbors_iter_(Graph self,int nidx,obj=False,data=False):
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
			
		return NeighborIterator(self,nidx,obj,data,False)
			
	cpdef grp_neighbors_iter(Graph self,nbunch,data=False):
		"""
		Return an iterator over a group of nodes' immediate neighbors.

		By default, the iterator will yield the node object for each immediate neighbor of nodes in ``nbunch``.

		**Args**:

			* ``nbunch``: an iterable providing the node object over whose neighbors to iterate.

			* ``data [=False]`` (boolean): if ``True``, then a tuple is yielded (rather than a node object) 
				containing the node	object and the data object associated with the node (e.g., ``(n,d)``).
		"""
		return SomeNeighborIterator(self,[self.node_idx_lookup[x] for x in nbunch],False,data,True)

	cpdef grp_neighbors_iter_(Graph self,nbunch,obj=False,data=False):
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
				
		return SomeNeighborIterator(self,nbunch,obj,data,False)
								
cdef class NodeIterator:
	cdef bint data
	cdef Graph graph
	cdef int idx
	cdef int node_count
	cdef bint nobj
	cdef bint obj
	cdef long init_num_changes
	
	def __cinit__(NodeIterator self,Graph graph,bint obj,bint data,bint nobj):
		self.init_num_changes = graph.num_changes
		self.data = data
		self.graph = graph
		self.idx = 0
		self.node_count = 0
		self.nobj = nobj
		self.obj = obj

	def __next__(NodeIterator self):		
		if self.node_count >= self.graph.num_nodes:
			raise StopIteration()

		if self.init_num_changes != self.graph.num_changes:
			raise GraphChangedException()

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
	cdef Graph graph
	cdef int idx
	cdef bint endpoints
	cdef long init_num_changes
	
	def __cinit__(AllEdgeIterator self,Graph graph,weight=False,data=False,endpoints=False):
		self.init_num_changes = graph.num_changes
		self.graph = graph
		self.data = data
		self.weight = weight
		self.idx = 0
		self.endpoints = endpoints
		
	def __next__(AllEdgeIterator self):
		cdef int idx = self.idx
		
		if self.init_num_changes != self.graph.num_changes:
			raise GraphChangedException()
			
		if idx == self.graph.next_edge_idx:
			raise StopIteration()
		
		while idx < self.graph.next_edge_idx and not self.graph.edge_info[idx].exists:
			idx += 1

		if idx >= self.graph.next_edge_idx:
			self.idx = idx
			raise StopIteration()
			
		self.idx = idx + 1
		if self.data and self.weight:
			val = None
			if idx in self.graph.edge_data_lookup:
				val = self.graph.edge_data_lookup[idx]
			if not self.endpoints:
				return idx, val, self.graph.edge_info[idx].weight
			else:
				if self.graph.edge_info[idx].u not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[idx].u
				elif self.graph.edge_info[idx].v not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[idx].v
					
				return self.graph.node_obj_lookup[self.graph.edge_info[idx].u], self.graph.node_obj_lookup[self.graph.edge_info[idx].v], val, self.graph.edge_info[idx].weight
		elif self.data is True:
			val = None
			if idx in self.graph.edge_data_lookup:
				val = self.graph.edge_data_lookup[idx]
			if not self.endpoints:
				return idx, val
			else:
				if self.graph.edge_info[idx].u not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[idx].u
				elif self.graph.edge_info[idx].v not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[idx].v
					
				return self.graph.node_obj_lookup[self.graph.edge_info[idx].u], self.graph.node_obj_lookup[self.graph.edge_info[idx].v], val
		elif self.weight:
			if not self.endpoints:
				return idx, self.graph.edge_info[idx].weight
			else:
				if self.graph.edge_info[idx].u not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[idx].u
				elif self.graph.edge_info[idx].v not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[idx].v
					
				return self.graph.node_obj_lookup[self.graph.edge_info[idx].u], self.graph.node_obj_lookup[self.graph.edge_info[idx].v], self.graph.edge_info[idx].weight
		else:
			if not self.endpoints:
				return idx
			else:
				if self.graph.edge_info[idx].u not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[idx].u
				elif self.graph.edge_info[idx].v not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[idx].v
					
				return self.graph.node_obj_lookup[self.graph.edge_info[idx].u], self.graph.node_obj_lookup[self.graph.edge_info[idx].v]

	def __iter__(AllEdgeIterator self):
		return self
		
cdef class NodeEdgeIterator:
	cdef bint data
	cdef bint weight
	cdef Graph graph
	cdef int nidx
	cdef int deg
	cdef int idx
	cdef bint endpoints
	cdef long init_num_changes
	
	def __cinit__(NodeEdgeIterator self,Graph graph,nidx,weight=False,data=False,endpoints=False):
		self.init_num_changes = graph.num_changes
		self.graph = graph
		self.nidx = nidx
		self.data = data
		self.weight = weight
		self.idx = 0
		self.endpoints = endpoints
			
	def __next__(NodeEdgeIterator self):
		if self.init_num_changes != self.graph.num_changes:
			raise GraphChangedException()
			
		cdef int idx = self.idx
		cdef int* elist
		
		num_edges = self.graph.node_info[self.nidx].degree
		elist = self.graph.node_info[self.nidx].elist

		if idx >= num_edges:
			raise StopIteration

		self.idx = idx + 1
		
		if self.data and self.weight:
			val = None
			if elist[idx] in self.graph.edge_data_lookup:
				val = self.graph.edge_data_lookup[elist[idx]], self.graph.edge_info[elist[idx]].weight
			if not self.endpoints:
				return elist[idx], val
			else:
				idx = elist[idx]
				if self.graph.edge_info[idx].u not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[idx].u
				elif self.graph.edge_info[idx].v not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[idx].v
					
				return self.graph.node_obj_lookup[self.graph.edge_info[idx].u], self.graph.node_obj_lookup[self.graph.edge_info[idx].v], val, self.graph.edge_info[idx].weight
		elif self.data is True:
			val = None
			if elist[idx] in self.graph.edge_data_lookup:
				val = self.graph.edge_data_lookup[elist[idx]]
			if not self.endpoints:
				return elist[idx], val
			else:
				idx = elist[idx]
				if self.graph.edge_info[idx].u not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[idx].u
				elif self.graph.edge_info[idx].v not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[idx].v
					
				return self.graph.node_obj_lookup[self.graph.edge_info[idx].u], self.graph.node_obj_lookup[self.graph.edge_info[idx].v], val
		elif self.weight:
			if not self.endpoints:
				return elist[idx], self.graph.edge_info[elist[idx]].weight
			else:
				idx = elist[idx]
				if self.graph.edge_info[idx].u not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[idx].u
				elif self.graph.edge_info[idx].v not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[idx].v
					
				return self.graph.node_obj_lookup[self.graph.edge_info[idx].u], self.graph.node_obj_lookup[self.graph.edge_info[idx].v], self.graph.edge_info[idx].weight
		else:
			if not self.endpoints:
				return elist[idx]
			else:
				idx = elist[idx]
				if self.graph.edge_info[idx].u not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[idx].u
				elif self.graph.edge_info[idx].v not in self.graph.node_obj_lookup:
					raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[idx].v
					
				return self.graph.node_obj_lookup[self.graph.edge_info[idx].u], self.graph.node_obj_lookup[self.graph.edge_info[idx].v]

	def __iter__(NodeEdgeIterator self):
		return self

cdef class SomeEdgeIterator:
	cdef bint data
	cdef bint weight
	cdef Graph graph
	cdef touched_edges
	cdef nbunch_iter
	cdef edge_iter
	cdef bint endpoints
	cdef long init_num_changes
	
	def __cinit__(SomeEdgeIterator self,Graph graph,nbunch,weight=False,data=False,endpoints=False):
		self.init_num_changes = graph.num_changes
		self.graph = graph
		self.nbunch_iter = iter(nbunch)
		self.data = data
		self.weight = weight
		self.edge_iter = None
		self.touched_edges = set()
		self.endpoints = endpoints
		
		# setup the first iterator
		if len(nbunch) > 0:
			curr_nidx = self.nbunch_iter.next()
			self.edge_iter = NodeEdgeIterator(self.graph,curr_nidx,self.weight,self.data)
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
					if self.data and self.weight:
						if result[0] in self.touched_edges:
							continue
						self.touched_edges.add(result[0])
						
						if not self.endpoints:
							return result
						else:
							if self.graph.edge_info[result[0]].u not in self.graph.node_obj_lookup:
								raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[result[0]].u
							elif self.graph.edge_info[result[0]].v not in self.graph.node_obj_lookup:
								raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[result[0]].v
								
							return self.graph.node_obj_lookup[self.graph.edge_info[result[0]].u], self.graph.node_obj_lookup[self.graph.edge_info[result[0]].v], result[1], result[2]
					elif self.data or self.weight:
						if result[0] in self.touched_edges:
							continue
						self.touched_edges.add(result[0])
						
						if not self.endpoints:
							return result
						else:
							if self.graph.edge_info[result[0]].u not in self.graph.node_obj_lookup:
								raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[result[0]].u
							elif self.graph.edge_info[result[0]].v not in self.graph.node_obj_lookup:
								raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[result[0]].v
								
							return self.graph.node_obj_lookup[self.graph.edge_info[result[0]].u], self.graph.node_obj_lookup[self.graph.edge_info[result[0]].v], result[1]
					else:
						if result in self.touched_edges:
							continue
						self.touched_edges.add(result)
					
						if not self.endpoints:
							return result
						else:
							if self.graph.edge_info[result].u not in self.graph.node_obj_lookup:
								raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[result].u
							elif self.graph.edge_info[result].v not in self.graph.node_obj_lookup:
								raise ZenException, 'Missing endpoint: Node (idx=%d) does not have an object' % self.graph.edge_info[result].v
								
							return self.graph.node_obj_lookup[self.graph.edge_info[result].u], self.graph.node_obj_lookup[self.graph.edge_info[result].v]
				except StopIteration:
					self.edge_iter = None
					curr_nidx = self.nbunch_iter.next()
					self.edge_iter = NodeEdgeIterator(self.graph,curr_nidx,self.weight,self.data)

	def __iter__(SomeEdgeIterator self):
		return self
						
cdef class NeighborIterator:
	cdef NodeEdgeIterator inner_iter
	cdef bint data
	cdef int nidx
	cdef Graph G
	cdef touched_nodes
	cdef bint use_nobjs
	cdef bint obj
	cdef long init_num_changes
	
	def __cinit__(NeighborIterator self, Graph G, int nidx,obj,data,use_nobjs):
		self.init_num_changes = G.num_changes
		self.inner_iter = NodeEdgeIterator(G,nidx,False)
		self.data = data
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
				if self.nidx == self.G.edge_info[eid].u:
					if self.G.edge_info[eid].v in self.touched_nodes:
						continue
					self.touched_nodes.add(self.G.edge_info[eid].v)
				
					if self.use_nobjs:
						if self.G.edge_info[eid].v not in self.G.node_obj_lookup:
							raise ZenException, 'Node (idx=%d) does not have a node object' % self.G.edge_info[eid].v
						return self.G.node_obj_lookup[self.G.edge_info[eid].v]
					else:
						return self.G.edge_info[eid].v
				else:
					if self.G.edge_info[eid].u in self.touched_nodes:
						continue
					self.touched_nodes.add(self.G.edge_info[eid].u)
					
					if self.use_nobjs:
						if self.G.edge_info[eid].u not in self.G.node_obj_lookup:
							raise ZenException, 'Node (idx=%d) does not have a node object' % self.G.edge_info[eid].u
						return self.G.node_obj_lookup[self.G.edge_info[eid].u]
					else:
						return self.G.edge_info[eid].u
			elif self.obj and not self.data:
				val = None
				if self.nidx == self.G.edge_info[eid].u:
					if self.G.edge_info[eid].v in self.touched_nodes:
						continue
					self.touched_nodes.add(self.G.edge_info[eid].v)	
				
					if self.use_nobjs:
						if self.G.edge_info[eid].v not in self.G.node_obj_lookup:
							raise ZenException, 'Node (idx=%d) does not have a node object' % self.G.edge_info[eid].v
						return self.G.node_obj_lookup[self.G.edge_info[eid].v], self.G.node_object(self.G.edge_info[eid].v)
					else:
						return self.G.edge_info[eid].v, self.G.node_object(self.G.edge_info[eid].v)
				else:
					if self.G.edge_info[eid].u in self.touched_nodes:
						continue
					self.touched_nodes.add(self.G.edge_info[eid].u)
					
					if self.use_nobjs:
						if self.G.edge_info[eid].u not in self.G.node_obj_lookup:
							raise ZenException, 'Node (idx=%d) does not have a node object' % self.G.edge_info[eid].u
						return self.G.node_obj_lookup[self.G.edge_info[eid].u], self.G.node_object(self.G.edge_info[eid].u)
					else:
						return self.G.edge_info[eid].u, self.G.node_object(self.G.edge_info[eid].u)
			elif not self.obj and self.data:
				val = None
				if self.nidx == self.G.edge_info[eid].u:
					if self.G.edge_info[eid].v in self.touched_nodes:
						continue
					self.touched_nodes.add(self.G.edge_info[eid].v)	
				
					if self.use_nobjs:
						if self.G.edge_info[eid].v not in self.G.node_obj_lookup:
							raise ZenException, 'Node (idx=%d) does not have a node object' % self.G.edge_info[eid].v
						return self.G.node_obj_lookup[self.G.edge_info[eid].v], self.G.node_data_(self.G.edge_info[eid].v)
					else:
						return self.G.edge_info[eid].v, self.G.node_data_(self.G.edge_info[eid].v)
				else:
					if self.G.edge_info[eid].u in self.touched_nodes:
						continue
					self.touched_nodes.add(self.G.edge_info[eid].u)
					
					if self.use_nobjs:
						if self.G.edge_info[eid].u not in self.G.node_obj_lookup:
							raise ZenException, 'Node (idx=%d) does not have a node object' % self.G.edge_info[eid].u
						return self.G.node_obj_lookup[self.G.edge_info[eid].u], self.G.node_data_(self.G.edge_info[eid].u)
					else:
						return self.G.edge_info[eid].u, self.G.node_data_(self.G.edge_info[eid].u)
			else:
				if self.nidx == self.G.edge_info[eid].u:
					if self.G.edge_info[eid].v in self.touched_nodes:
						continue
					self.touched_nodes.add(self.G.edge_info[eid].v)	
				
					if self.use_nobjs:
						if self.G.edge_info[eid].v not in self.G.node_obj_lookup:
							raise ZenException, 'Node (idx=%d) does not have a node object' % self.G.edge_info[eid].v
						return self.G.node_obj_lookup[self.G.edge_info[eid].v], self.G.node_object(self.G.edge_info[eid].v), self.G.node_data_(self.G.edge_info[eid].v)
					else:
						return self.G.edge_info[eid].v, self.G.node_object(self.G.edge_info[eid].v), self.G.node_data_(self.G.edge_info[eid].v)
				else:
					if self.G.edge_info[eid].u in self.touched_nodes:
						continue
					self.touched_nodes.add(self.G.edge_info[eid].u)
					
					if self.use_nobjs:
						if self.G.edge_info[eid].u not in self.G.node_obj_lookup:
							raise ZenException, 'Node (idx=%d) does not have a node object' % self.G.edge_info[eid].u
						return self.G.node_obj_lookup[self.G.edge_info[eid].u], self.G.node_object(self.G.edge_info[eid].u), self.G.node_data_(self.G.edge_info[eid].u)
					else:
						return self.G.edge_info[eid].u, self.G.node_object(self.G.edge_info[eid].u), self.G.node_data_(self.G.edge_info[eid].u)
		
	def __iter__(NeighborIterator self):
		return self
		
cdef class SomeNeighborIterator:
	cdef bint data
	cdef Graph graph
	cdef int idx
	cdef touched_nodes
	cdef nbunch_iter
	cdef neighbor_iter
	cdef bint use_nobjs
	cdef bint obj
	cdef long init_num_changes

	def __cinit__(SomeNeighborIterator self,Graph graph,nbunch,obj,data,use_nobjs):
		self.init_num_changes = graph.num_changes
		self.graph = graph
		self.nbunch_iter = iter(nbunch)
		self.obj = obj
		self.data = data
		self.idx = 0
		self.neighbor_iter = None
		self.touched_nodes = set()
		self.use_nobjs = use_nobjs
		
		# setup the first iterator
		if len(nbunch) > 0:
			curr_nidx = self.nbunch_iter.next()
			self.neighbor_iter = NeighborIterator(self.graph,curr_nidx,self.obj,self.data,self.use_nobjs)
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
					self.neighbor_iter = NeighborIterator(self.graph,curr_nidx,self.obj,self.data,self.use_nobjs)

	def __iter__(SomeNeighborIterator self):
		return self
