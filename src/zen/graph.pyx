#cython: embedsignature=True

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

__all__ = ['Graph']

cdef struct NodeInfo:
	# This data structure contains node info (in C-struct format) for fast array-based lookup.
	bint exists
	
	int degree # The number of entries in the edge list that are in use
	int* elist
	int capacity  # The length of the edge list
	
cdef int sizeof_NodeInfo = sizeof(NodeInfo)

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

cdef class Graph:
	"""
	This class provides a highly-optimized implementation of an undirected graph.  Duplicate edges are not allowed.
	
	Public properties include:
		- max_node_index - the largest node index currently in use
		- max_edge_index - the largest edge index currently in use
	"""
	
	def __cinit__(Graph self,node_capacity=100,edge_capacity=100,edge_list_capacity=5):
		"""
		Initialize the graph.
		
		  node_capacity is the initial number of nodes this graph has space to hold
		  edge_capacity is the initial number of edges this graph has space to hold
		  edge_list_capacity is the initial number of edges that each node is allocated space for initially.
		
		"""
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
		self.node_info = <NodeInfo*> stdlib.malloc(sizeof_NodeInfo*self.node_capacity)
		for i in range(self.node_capacity):
			self.node_info[i].exists = False
		self.node_obj_lookup = {}
		self.node_data_lookup = {}
		self.node_idx_lookup = {}
		
		self.num_edges = 0
		self.edge_capacity = edge_capacity
		self.next_edge_idx = 0
		self.edge_info = <EdgeInfo*> stdlib.malloc(sizeof_EdgeInfo*self.edge_capacity)
		for i in range(self.edge_capacity):
			self.edge_info[i].exists = False
		self.edge_data_lookup = {}
		
		self.edge_list_capacity = edge_list_capacity
	
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
								self.node_info[i].degree)
				
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
				exists, degree = entry
				self.node_info[i].exists = exists
				self.node_info[i].degree = degree
				self.node_info[i].capacity = -1
				
		# restore all edge details
		self.num_edges = state['num_edges']
		self.edge_capacity = state['edge_capacity']
		self.edge_grow_factor = state['edge_grow_factor']
		self.next_edge_idx = state['next_edge_idx']
		self.first_free_edge = state['first_free_edge']
		self.edge_data_lookup = state['edge_data_lookup']

		# restore edge_info
		self.edge_info = <EdgeInfo*> stdlib.malloc(sizeof_EdgeInfo*self.edge_capacity)
		for i,entry in enumerate(state['edge_info']):
			if entry[0] is True:
				exists,u,v,weight = entry
				self.edge_info[i].exists = exists
				self.edge_info[i].u = u
				self.edge_info[i].v = v
				self.edge_info[i].weight = weight
			else:
				self.edge_info[i].exists = exists
				
		return
	
	def __getattr__(self,name):
		
		if name == 'max_node_idx':
			return self.next_node_idx - 1
		elif name == 'max_edge_idx':
			return self.next_edge_idx - 1
		else:
			raise AttributeError, 'Class has no attribute "%s"' % name
	
	cpdef copy(Graph self):
		"""
		Create a copy of this graph.
		
		Note that node and edge indices are preserved in this copy.
		"""
		cdef Graph G = Graph()
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
		
	cpdef is_directed(Graph self):
		"""
		Return True if this graph is directed (which it is not).
		"""
		return False
	
	cpdef bool is_compact(Graph self):
		"""
		Return True if the graph is currently in compact form: there are no unallocated node indices i < self.max_node_idx and no
		unallocated edge indices j < self.max_edge_idx.  The graph can be compacted by calling the compact() method.
		"""
		return (self.first_free_node == -1 and self.first_free_edge == -1)
	
	cpdef compact(Graph self):
		"""
		Compact the graph in place.  This will re-assign:
		
			1) node indices such that there are no unallocated node indices less than self.max_node_idx
			2) edge indices such that there are no unallocated edge indices less than self.max_edge_idx
			
		Note that at present no way is provided of keeping track of the changes made to node and edge indices.
		"""
		cdef int next_free_idx
		cdef int src, dest
		cdef int u,v,i
		cdef int eidx
		
		#####
		# move the nodes around
		while self.first_free_node != -1:
			next_free_idx = self.node_info[self.first_free_node].degree
			
			# move all node content 
			src = self.next_node_idx - 1
			dest = self.first_free_node
			self.node_info[dest] = self.node_info[src]
			self.node_info[src].exists = False
			self.next_node_idx -= 1
			
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
								
			# move to the next one
			self.first_free_node = next_free_idx
			
			if self.first_free_node == (self.next_node_idx - 1):
				self.first_free_node = -1
				self.next_node_idx -= 1
				
		#####
		# move the edges around
		while self.first_free_edge != -1:
			next_free_idx = self.edge_info[self.first_free_edge].u
			
			# move all edge content from the last element to the first free edge
			src = self.next_edge_idx - 1
			dest = self.first_free_edge
			self.edge_info[dest] = self.edge_info[src]
			self.edge_info[src].exists = False
			self.next_edge_idx -= 1
			
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
			
			# move to the next one
			self.first_free_edge = next_free_idx
			
			if self.first_free_edge == (self.next_edge_idx - 1):
				self.first_free_edge = -1
				self.next_edge_idx -= 1
		
		return
	
	cpdef np.ndarray[np.double_t] matrix(self):
		"""
		Return a numpy adjacency matrix.
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
		
		If node_obj_fxn is specified, then this function will be called with each node index
		and the object returned will be used as the node object for that node.  If not specified,
		then no node objects will be assigned to nodes.
		"""
		cdef int i
		cdef int nn_count
		cdef int num_remaining
		cdef int new_node_capacity
		cdef np.ndarray[np.int_t,ndim=1] indexes = np.empty(num_nodes, np.int)
		cdef int node_idx
		
		self.num_changes += 1
		
		for nn_count in range(num_nodes):
			
			# recycle an unused node index if possible
			if self.first_free_node != -1:
				node_idx = self.first_free_node
				self.first_free_node = self.node_info[node_idx].degree
			else:
				node_idx = self.next_node_idx
				self.next_node_idx += 1

			indexes[nn_count] = node_idx
			
			# set the node object if specified
			if node_obj_fxn is not None:
				nobj = node_obj_fxn(node_idx)
				self.node_idx_lookup[nobj] = node_idx
				self.node_obj_lookup[node_idx] = nobj
		
			# grow the node_info array as necessary
			if node_idx >= self.node_capacity:
				# allocate exactly as many nodes as are needed
				num_remaining = num_nodes - nn_count + 1
				new_node_capacity = self.node_capacity + num_remaining
				self.node_info = <NodeInfo*> stdlib.realloc(self.node_info, sizeof_NodeInfo*new_node_capacity)

				for i in range(self.node_capacity,new_node_capacity):
					self.node_info[i].exists = False
				self.node_capacity = new_node_capacity
			
			self.node_info[node_idx].exists = True
			self.node_info[node_idx].degree = 0
		
			# initialize edge lists
			self.node_info[node_idx].elist = <int*> stdlib.malloc(sizeof(int) * self.edge_list_capacity)
			self.node_info[node_idx].capacity = self.edge_list_capacity
		
			self.num_nodes += 1
			
		return indexes
		
	cpdef int add_node(Graph self,nobj=None,data=None):
		"""
		Add a node to this graph.
		
		  nobj is an optional node object
		  data is an optional data object to associated with this node
		
		This method returns the index corresponding to the new node.
		"""
		cdef int node_idx
		
		# recycle an unused node index if possible
		if self.first_free_node != -1:
			node_idx = self.first_free_node
			self.first_free_node = self.node_info[node_idx].degree
		else:
			node_idx = self.next_node_idx
			self.next_node_idx += 1
			
		self.add_node_x(node_idx,self.edge_list_capacity,nobj,data)
		
		return node_idx
		
	cpdef add_node_x(Graph self,int node_idx,int edge_list_capacity,nobj,data):
		"""
		This function adds a node to the graph, but should be used with GREAT CARE.
		
		This function permits very high-performance population of the graph data structure
		with nodes by allowing the calling function to specify the node index and edge
		capacity of the node being added.  In general, this should only be done when the node indices
		have been obtained from a previously stored graph data structure.
		
		Needless to say, when used incorrectly, this method call can irreparably damage
		the integrity of the graph object, leading to incorrect results or, more likely,
		segmentation faults.
		"""
		cdef int i
		
		if node_idx >= self.next_node_idx:
			self.next_node_idx = node_idx + 1
		
		self.num_changes += 1
		
		if nobj is not None:
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
			self.node_capacity = new_node_capacity
			
		self.node_info[node_idx].exists = True
		self.node_info[node_idx].degree = 0
		
		# initialize edge lists
		self.node_info[node_idx].elist = <int*> stdlib.malloc(sizeof(int) * self.edge_list_capacity)
		self.node_info[node_idx].capacity = self.edge_list_capacity
		
		self.num_nodes += 1
		return
		
	def __contains__(Graph self,nobj):
		"""
		Return True if the node object is in the graph.
		"""
		return nobj in self.node_idx_lookup
	
	cpdef int node_idx(Graph self,nobj) except -1:
		"""
		Return the node index associated with the node object.
		
		If the node object is not in the graph, an exception is raised.
		"""
		return self.node_idx_lookup[nobj]
		
	cpdef node_object(Graph self,int nidx):
		"""
		Return the node object associated with the node index.
		
		If no node object is associated, then None is returned.
		"""
		if nidx in self.node_obj_lookup:
			return self.node_obj_lookup[nidx]
		else:
			return None
	
	cpdef set_node_data(Graph self,nobj,data):
		"""
		Associate a new data object with a specific node in the network.
		If data is None, then any data associated with the node is deleted.
		"""
		self.set_node_data_(self.node_idx_lookup[nobj],data)

	cpdef set_node_data_(Graph self,int nidx,data):
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx
			
		if data == None:
			if nidx in self.node_data_lookup:
				del self.node_data_lookup[nidx]
		else:
			self.node_data_lookup[nidx] = data
						
	cpdef node_data(Graph self,nobj):
		"""
		Return the data object that is associated with the node object.
		
		If the node object is not in the graph, then an Exception is raised.
		"""
		return self.node_data_(self.node_idx_lookup[nobj])

	cpdef node_data_(Graph self,int nidx):
		"""
		Return the data object that is associated with the node index.
		
		If no data object is associated, then None is returned.
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
		
		If data is False, the iterator returns node objects.  If data is True,
		then the iterator returns tuples of (node object,node data).
		"""
		return NodeIterator(self,False,data,True)

	cpdef nodes_iter_(Graph self,obj=False,data=False):
		"""
		Return an iterator over all the nodes in the graph.
	
		If obj and data are False, the iterator returns node indices.  
		If obj and/or data is True, then the iterator returns tuples of (node index [,node object] [,node data]).
		"""
		return NodeIterator(self,obj,data,False)
				
	cpdef nodes(Graph self,data=False):
		"""
		Return a list of nodes.  If data is False, then the result is a
		list of the node objects.  If data is True, then the result is list of
		tuples containing the node object and associated data.
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
		Return a numpy array of the nodes.  If obj and data are False, then the result is a
		1-D array of the node indices.  If obj or data is True, then the result is a 2-D 
		array in which the first column is node indices and the second column is
		the node obj/data.  If obj and data are True, then the result is a 2-D array
		in which the first column node indices, the second column is the node object, and the
		third column is the node data.
		
		if obj and data are both False, then the numpy array returned has type int.  When used
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
		Remove the node associated with node object nobj.
		"""
		self.rm_node_(self.node_idx_lookup[nobj])
	
	cpdef rm_node_(Graph self,int nidx):
		"""
		Remove the node with index nidx.
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
			
		if nidx in self.node_obj_lookup:
			nobj = self.node_obj_lookup[nidx]
			del self.node_obj_lookup[nidx]
			del self.node_idx_lookup[nobj]
		
		# keep track of the free node
		if self.first_free_node == -1:
			self.first_free_node = nidx
			self.node_info[nidx].degree = -1
		else:
			self.node_info[nidx].degree = self.first_free_node
			self.first_free_node = nidx
						
		self.num_nodes -= 1
	
	cpdef degree(Graph self,nobj):
		"""
		Return the degree of node with object nobj.
		"""
		return self.degree_(self.node_idx_lookup[nobj])

	cpdef degree_(Graph self,int nidx):
		"""
		Return the degree of node with index nidx.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx
			
		return self.node_info[nidx].degree
	
	def __getitem__(self,nobj):
		"""
		Get the data for the node with the object given
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
	
	# def compact(self):
	# 	"""
	# 	Re-index edges and nodes in order to recapture wasted space in the
	# 	graph data structure.  After calling this, node indices and edge
	# 	indices may be different.
	# 	
	# 	TODO: Not implemented
	# 	"""
	# 	raise Exception, 'NOT IMPLEMENTED'
	
	cpdef int add_edge(self, u, v, data=None, double weight=1) except -1:
		"""
		Add an edge to the graph from the src node to the tgt node.
		src and tgt are node objects.  If data is not None, then it
		is used as the data associated with this edge.
		
		This function returns the index for the new edge.
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
		Add an edge to the graph connecting nodes with indices u and v.
		If data is not None, then it is used as the data associated with this edge.
	
		This function returns the index for the new edge.
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
			self.first_free_edge = self.edge_info[eidx].u
		else:
			eidx = self.next_edge_idx
			self.next_edge_idx += 1
	
		self.add_edge_x(eidx,u,v,data,weight)

		return eidx

	cpdef int add_edge_x(self, int eidx, int u, int v, data, double weight) except -1:

		if eidx >= self.next_edge_idx:
			self.next_edge_idx = eidx + 1
				
		if data is not None:
			self.edge_data_lookup[eidx] = data
		
		# grow the info array
		cdef int new_edge_capacity
		if eidx >= self.edge_capacity:
			new_edge_capacity = <int> ceil( <float>self.edge_capacity * self.edge_grow_factor)
			self.edge_info = <EdgeInfo*> stdlib.realloc( self.edge_info, sizeof_EdgeInfo * new_edge_capacity)
			for i in range(self.edge_capacity,new_edge_capacity):
				self.edge_info[i].exists = False
			self.edge_capacity = new_edge_capacity
		
		####
		# connect up the edges to nodes
		
		# u
		self.__insert_edge_into_edgelist(u,eidx,v)
		
		# v
		self.__insert_edge_into_edgelist(v,eidx,u)
		
		### Add edge info
		self.edge_info[eidx].exists = True
		self.edge_info[eidx].u = u
		self.edge_info[eidx].v = v
		self.edge_info[eidx].weight = weight
		
		#####
		# Done
		self.num_edges += 1
		return eidx
	
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
		Remove the edge between node objects u and v.
		"""
		self.rm_edge_(self.edge_idx(u,v))
	
	cpdef rm_edge_(Graph self,int eidx):
		"""
		Remove the edge with index eidx.
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
		if self.first_free_edge == -1:
			self.first_free_edge = eidx
			self.edge_info[eidx].u = -1
		else:
			self.edge_info[eidx].u = self.first_free_edge
			self.first_free_edge = eidx
		
		self.num_edges -= 1
		
		return
	
	cdef __remove_edge_from_edgelist(Graph self, int u, int eidx, int v):
		cdef int i = self.find_elist_insert_pos(self.node_info[u].elist,self.node_info[u].degree,u,v)
		memmove(&self.node_info[u].elist[i],&self.node_info[u].elist[i+1],(self.node_info[u].degree-i-1)*sizeof(int))
		self.node_info[u].degree -= 1
		
		return
	
	cpdef endpoints(Graph self,int eidx):
		"""
		Return the node objects at the endpoints of the edge with index eidx.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Edge with ID %d does not exist' % eidx
	
		return self.node_obj_lookup[self.edge_info[eidx].u], self.node_obj_lookup[self.edge_info[eidx].v]
		
	cpdef endpoints_(Graph self,int eidx):
		"""
		Return the node indices at the endpoints of the edge with index eidx.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx

		return self.edge_info[eidx].u, self.edge_info[eidx].v
	
	cpdef endpoint(Graph self,int eidx,u):
		"""
		Return the other node (not u) that is the endpoint of this edge.  Note, no check is done
		to ensure that u is an endpoint of the edge.
		"""
		if u not in self.node_idx_lookup:
			raise ZenException, 'Invalid node object %s' % str(u)
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx

		return self.node_object(self.endpoint_(eidx,self.node_idx_lookup[u]))
	
	cpdef int endpoint_(Graph self,int eidx,int u) except -1:
		"""
		Return the other endpoint for edge eidx besides the one given (u).
		
		Note that this method is implemented for speed and no check is made to ensure that
		u is one of the edge's endpoints.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx
			
		if u == self.edge_info[eidx].u:
			return self.edge_info[eidx].v
		else:
			return self.edge_info[eidx].u
		
	cpdef set_weight(Graph self,u,v,double w):
		self.set_weight_(self.edge_idx(u,v),w)

	cpdef set_weight_(Graph self,int eidx,double w):
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx
		
		self.edge_info[eidx].weight = w
		
	cpdef double weight(Graph self,u,v):
		return self.weight_(self.edge_idx(u,v))

	cpdef double weight_(Graph self,int eidx):
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx
			
		return self.edge_info[eidx].weight
		
	cpdef set_edge_data(Graph self,u,v,data):
		"""
		Associate a new data object with a specific edge in the network.
		If data is None, then any data associated with the edge is deleted.
		"""
		self.set_edge_data_(self.edge_idx(u,v),data)
		
	cpdef edge_data(Graph self,u,v):
		"""
		Return the data associated with the edge connecting u and v.
		"""
		return self.edge_data_(self.edge_idx(u,v))
		
	cpdef set_edge_data_(Graph self,int eidx,data):
		"""
		Associate a new data object with a specific edge in the network.
		If data is None, then any data associated with the edge is deleted.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx
			
		if data == None:
			if eidx in self.edge_data_lookup:
				del self.edge_data_lookup[eidx]
		else:
			self.edge_data_lookup[eidx] = data
		
	cpdef edge_data_(Graph self,int eidx,int v=-1):
		"""
		Return the data associated with the edge with index eidx.
		
		If v is specified, then the data associated with the edge
		connecting nodes with indices eidx and v is returned.
		"""
		if v != -1:
			eidx = self.edge_idx_(eidx,v)			
		elif eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx
	
		if eidx in self.edge_data_lookup:
			return self.edge_data_lookup[eidx]
		else:
			return None
			
	cpdef bool has_edge(Graph self,u,v):
		"""
		Return True if the graph contains an edge connecting node objects u and v.
		
		If u or v are not in the graph, this method returns False.
		"""
		if u not in self.node_idx_lookup:
			return False
		if v not in self.node_idx_lookup:
			return False
			
		u = self.node_idx_lookup[u]
		v = self.node_idx_lookup[v]
		
		return self.has_edge_(u,v)
		
	cpdef bool has_edge_(Graph self,int u,int v):
		"""
		Return True if the graph contains an edge connecting nodes with indices u and v.
		
		Both u and v must be valid node indices.
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
	
	cpdef edge_idx(Graph self, u, v, data=False):
		"""
		Return the edge index for the edge connecting node objects u and v.
		"""
		u = self.node_idx_lookup[u]
		v = self.node_idx_lookup[v]
		return self.edge_idx_(u,v,data)
	
	cpdef edge_idx_(Graph self, int u, int v, data=False):
		"""
		Return the edge index for the edge connecting node indices u and v.
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
			if data is True:
				return elist[pos], self.edge_data_lookup[elist[pos]]
			else:
				return elist[pos]
		else:			
			raise ZenException, 'Edge (%d,%d) does not exist.' % (u,v)
	
	cpdef edges_iter(Graph self,nobj=None,bool data=False,bool weight=False):
		"""
		Return an iterator over edges in the graph.
		
		If nobj is None, then all edges in the graph are iterated over.  Otherwise
		the edges touching the node with object nobj are iterated over.
		
		If data is False, then the iterator returns a tuple (src,tgt).  Otherwise, the
		iterator returns a tuple (src,tgt,data).
		"""
		if nobj is None:
			return AllEdgeIterator(self,weight,data,True)
		else:
			return NodeEdgeIterator(self,self.node_idx_lookup[nobj],weight,data,True)
	
	cpdef edges_iter_(Graph self,int nidx=-1,bool data=False,bool weight=False):
		"""
		Return an iterator over edges in the graph.
	
		If nidx is None, then all edges in the graph are iterated over.  Otherwise
		the edges touching the node with index nidx are iterated over.
	
		If data is False, then the iterator returns edge indices.  Otherwise, the
		iterator returns a tuple (edge index,data).
		"""	
		if nidx == -1:
			return AllEdgeIterator(self,weight,data,False)
		else:
			if nidx >= self.node_capacity or not self.node_info[nidx].exists:
				raise ZenException, 'Invalid node idx %d' % nidx
				
			return NodeEdgeIterator(self,nidx,weight,data,False)
	
	cpdef edges(Graph self,nobj=None,bool data=False,bool weight=False):
		"""
		Return edges connected to a node.  If nobj is not specified, then
		all edges in the network are returned.
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
				
	cpdef edges_(Graph self,int nidx=-1,bool data=False,bool weight=False):
		"""
		Return a numpy array of edges.  If nidx is None, then all edges will be returned.
		If nidx is not None, then the edges for the node with nidx will be returned.
		
		If data is False, then the numpy array will be a 1-D array containing edge indices.  If data
		is True, then the numpy array will be a 2-D array containing indices in the first column and
		descriptor objects in the second column.
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
	
	cpdef grp_edges_iter(Graph self,nbunch,bool data=False,bool weight=False):
		"""
		Return an iterator over the edges of nodes in nbunch.  If data is 
		True then tuples (src,tgt,data) are returned.
		"""
		return SomeEdgeIterator(self,[self.node_idx_lookup[x] for x in nbunch],weight,data,True)
		
	cpdef grp_edges_iter_(Graph self,nbunch,bool data=False,bool weight=False):
		"""
		Return an iterator over the edges of node indices in nbunch.  If data is 
		True then tuples (eidx,data) are returned.
		"""
		for nidx in nbunch:
			if nidx >= self.node_capacity or not self.node_info[nidx].exists:
				raise ZenException, 'Invalid node idx %d' % nidx
				
		return SomeEdgeIterator(self,nbunch,weight,data)
						
	cpdef neighbors(Graph self,nobj,data=False):
		"""
		Return a list of nodes that are neighbors of the node nobj.  If data is True, then a 
		list of tuples is returned, each tuple containing a neighbor node object and its data.
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
		Return a numpy array of node ids corresponding to all neighbors of the node with id nid.
		
		If obj and data are False, then the numpy array will be a 1-D array containing node indices.  If obj or data
		are True, then the numpy array will be a 2-D array containing indices in the first column and
		the node object/data object in the second column.  If both obj and data are True, then the numpy array will be
		a 2-D array containing indices in the first column, the node object in the second, and the node data in 
		the third column.
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
		Return an iterator over the neighbors of node with object nobj.  If data is 
		True then tuples (obj,data) are returned.
		"""
		return NeighborIterator(self,self.node_idx_lookup[nobj],False,data,True)
		
	cpdef neighbors_iter_(Graph self,int nidx,obj=False,data=False):
		"""
		Return an iterator over the neighbors of node with index nidx.  If obj is True, then
		tuples (nidx,obj) are returned.  If data is True then tuples (obj,data) are returned.
		If both are True, then tuples (nix,obj,data) are returned.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx
			
		return NeighborIterator(self,nidx,obj,data,False)
			
	cpdef grp_neighbors_iter(Graph self,nbunch,data=False):
		"""
		Return an iterator over the neighbors of nodes in nbunch.  If data is 
		True then tuples (nobj,data) are returned.
		"""
		return SomeNeighborIterator(self,[self.node_idx_lookup[x] for x in nbunch],False,data,True)

	cpdef grp_neighbors_iter_(Graph self,nbunch,obj=False,data=False):
		"""
		Return an iterator over the neighbors of nodes in nbunch.  If obj is True
		then tuples (nidx,nobj) are returned.  If data is True then tuples (nidx,data) 
		are returned.  If both are True, then tuples (nidx,nobj,data) are returned.
		"""
		for nidx in nbunch:
			if nidx >= self.node_capacity or not self.node_info[nidx].exists:
				raise ZenException, 'Invalid node idx %d' % nidx
				
		return SomeNeighborIterator(self,nbunch,obj,data,False)
								
cdef class NodeIterator:
	cdef bool data
	cdef Graph graph
	cdef int idx
	cdef int node_count
	cdef bool nobj
	cdef bool obj
	cdef long init_num_changes
	
	def __cinit__(NodeIterator self,Graph graph,bool obj,bool data,bool nobj):
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
	cdef bool data
	cdef bool weight
	cdef Graph graph
	cdef int idx
	cdef bool endpoints
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
	cdef bool data
	cdef bool weight
	cdef Graph graph
	cdef int nidx
	cdef int deg
	cdef int idx
	cdef bool endpoints
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
	cdef bool data
	cdef bool weight
	cdef Graph graph
	cdef touched_edges
	cdef nbunch_iter
	cdef edge_iter
	cdef bool endpoints
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
	cdef bool data
	cdef int nidx
	cdef Graph G
	cdef touched_nodes
	cdef bool use_nobjs
	cdef bool obj
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
	cdef bool data
	cdef Graph graph
	cdef int idx
	cdef touched_nodes
	cdef nbunch_iter
	cdef neighbor_iter
	cdef bool use_nobjs
	cdef bool obj
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