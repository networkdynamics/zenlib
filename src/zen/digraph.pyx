#cython: embedsignature=True

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

cdef class DiGraph:
	"""
	This class provides a highly-optimized implementation of a directed graph.  Duplicate edges are not allowed.
	
	Public properties include:
		- max_node_index - the largest node index currently in use
		- max_edge_index - the largest edge index currently in use
	"""
	
	def __cinit__(DiGraph self,node_capacity=100,edge_capacity=100,edge_list_capacity=5):
		"""
		Initialize the directed graph.
		
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
								self.node_info[i].indegree)

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
		for i,entry in enumerate(state['node_info']):

			if entry[0] is True:
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
				exists, indegree = entry
				self.node_info[i].exists = exists
				self.node_info[i].indegree = indegree
				
				self.node_info[i].in_capacity = -1
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
		for i,entry in enumerate(state['edge_info']):
			if entry[0] is True:
				exists,src,tgt,weight = entry
				self.edge_info[i].exists = exists
				self.edge_info[i].src = src
				self.edge_info[i].tgt = tgt
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
	
	cpdef np.ndarray[np.double_t] matrix(self):
		"""
		Return a numpy adjacency matrix.
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
		
		Note that node and edge indices are preserved.
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
				
	cpdef is_directed(DiGraph self):
		"""
		Return True if this graph is directed (which it is).
		"""
		return True
	
	cpdef bool is_compact(DiGraph self):
		"""
		Return True if the graph is currently in compact form: there are no unallocated node indices i < self.max_node_idx and no
		unallocated edge indices j < self.max_edge_idx.  The graph can be compacted by calling the compact() method.
		"""
		return (self.first_free_node == -1 and self.first_free_edge == -1)
	
	cpdef compact(DiGraph self):
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
			next_free_idx = self.node_info[self.first_free_node].indegree
			
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
								
			# move to the next one
			self.first_free_node = next_free_idx
			if self.first_free_node == (self.next_node_idx - 1):
				self.first_free_node = -1
				self.next_node_idx -= 1
				
		#####
		# move the edges around
		while self.first_free_edge != -1:
			next_free_idx = self.edge_info[self.first_free_edge].src
			
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
			
			# move to the next one
			self.first_free_edge = next_free_idx
			
			if self.first_free_edge == (self.next_edge_idx - 1):
				self.first_free_edge = -1
				self.next_edge_idx -= 1
		
		return
					
	cpdef skeleton(self,data_merge_fxn=NO_NONE_LIST_OF_DATA,weight_merge_fxn=AVG_OF_WEIGHTS):
		"""
		Create an undirected version of this graph.  Note that node indices will
		be preserved in the undirected graph that is returned.
		
		data_merge_fxn decides how the data objects associated with reciprocal
		edges will be combined into a data object for a single undirected edge.
		Valid values values are:
			
			- NO_NONE_LIST_OF_DATA: a list of the data objects if the both data
			  objects are not None.  Otherwise, the value will be set to None.
			- LIST_OF_DATA: a list of the data objects regardless of their values.
			- an arbitrary function merge(i,j,d1,d2) that returns a single object.
			  In this instance, d1 is the data associated with edge (i,j) and 
			  d2 is the data object associated with edge (j,i).  Note that i and j
			  are node indices, not objects.
		
		weight_merge_fxn decides how the values of reciprocal edges will be 
		combined into a single undirected edge.  Valid values are 
		
			- AVG_OF_WEIGHTS: average the two weights
			- MIN_OF_WEIGHTS: take the min value of the weights
			- MAX_OF_WEIGHTS: take the max value of the two weights
			- an arbitrary function merge(i,j,w1,w2) that returns a float. 
			  In this instance, w1 is the weight associated with edge (i,j) and 
			  w2 is the weight associated with edge (j,i). Note that i and j are 
			  node indices, not objects.
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
		Create a directed graph with identical content in which edges are reverse directions.
		
		Note that node and edge indices are preserved in the reversed graph.
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

		If node_obj_fxn is specified, then this function will be called with each node index
		and the object returned will be used as the node object for that node.  If not specified,
		then no node objects will be assigned to nodes.
		"""
		cdef int i
		cdef int nn_count
		cdef int num_remaining
		cdef int new_node_capacity
		cdef np.ndarray[np.int_t,ndim=1] indexes = np.empty( num_nodes, np.int)
		cdef int node_idx
		
		self.num_changes += 1

		for nn_count in range(num_nodes):

			# recycle an unused node index if possible
			if self.first_free_node != -1:
				node_idx = self.first_free_node
				self.first_free_node = self.node_info[node_idx].indegree
			else:
				node_idx = self.next_node_idx
				self.next_node_idx += 1

			indexes[nn_count] = node_idx

			# add a node object if specified
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
			self.node_info[node_idx].indegree = 0
			self.node_info[node_idx].outdegree = 0

			# initialize edge lists
			self.node_info[node_idx].inelist = <int*> stdlib.malloc(sizeof(int) * self.edge_list_capacity)
			self.node_info[node_idx].in_capacity = self.edge_list_capacity
			self.node_info[node_idx].outelist = <int*> stdlib.malloc(sizeof(int) * self.edge_list_capacity)
			self.node_info[node_idx].out_capacity = self.edge_list_capacity

			self.num_nodes += 1

		return indexes
					
	cpdef int add_node(DiGraph self,nobj=None,data=None):
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
			self.first_free_node = self.node_info[node_idx].indegree
		else:
			node_idx = self.next_node_idx
			self.next_node_idx += 1
			
		self.add_node_x(node_idx,self.edge_list_capacity,self.edge_list_capacity,nobj,data)
		
		return node_idx
		
	cpdef add_node_x(DiGraph self,int node_idx,int in_edge_list_capacity,int out_edge_list_capacity,nobj,data):
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
		self.node_info[node_idx].indegree = 0
		self.node_info[node_idx].outdegree = 0
		
		# initialize edge lists
		self.node_info[node_idx].inelist = <int*> stdlib.malloc(sizeof(int) * self.edge_list_capacity)
		self.node_info[node_idx].in_capacity = self.edge_list_capacity
		self.node_info[node_idx].outelist = <int*> stdlib.malloc(sizeof(int) * self.edge_list_capacity)
		self.node_info[node_idx].out_capacity = self.edge_list_capacity
		
		self.num_nodes += 1
		return
	
	def __contains__(DiGraph self,nobj):
		"""
		Return True if the node object is in the graph.
		"""
		return nobj in self.node_idx_lookup
	
	cpdef int node_idx(DiGraph self,nobj) except -1:
		"""
		Return the node index associated with the node object.
		
		If the node object is not in the graph, an exception is raised.
		"""
		return self.node_idx_lookup[nobj]
		
	cpdef node_object(DiGraph self,int nidx):
		"""
		Return the node object associated with the node index.
		
		If no node object is associated, then None is returned.
		"""
		if nidx in self.node_obj_lookup:
			return self.node_obj_lookup[nidx]
		else:
			return None
		
	cpdef node_data(DiGraph self,nobj):
		"""
		Return the data object that is associated with the node object.
		
		If the node object is not in the graph, then an Exception is raised.
		"""
		return self.node_data_(self.node_idx_lookup[nobj])

	cpdef set_node_data(DiGraph self,nobj,data):
		"""
		Associate a new data object with a specific node in the network.
		If data is None, then any data associated with the node is deleted.
		"""
		self.set_node_data_(self.node_idx_lookup[nobj],data)
		
	cpdef set_node_data_(DiGraph self,int nidx,data):
		"""
		Associate a new data object with a specific node in the network.
		If data is None, then any data associated with the node is deleted.
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
		Return the data object that is associated with the node index.
		
		If no data object is associated, then None is returned.
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
		
		If data is False, the iterator returns node objects.  If data is True,
		then the iterator returns tuples of (node object,node data).
		"""
		return NodeIterator(self,False,data,True)

	cpdef nodes_iter_(DiGraph self,obj=False,data=False):
		"""
		Return an iterator over all the nodes in the graph.
	
		If obj and data are False, the iterator returns node indices.  
		If obj and/or data is True, then the iterator returns tuples of (node index [,node object] [,node data]).
		"""
		return NodeIterator(self,obj,data,False)
				
	cpdef nodes(DiGraph self,data=False):
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

	cpdef nodes_(DiGraph self,obj=False,data=False):
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
		
	cpdef rm_node(DiGraph self,nobj):
		"""
		Remove the node associated with node object nobj.
		"""
		self.rm_node_(self.node_idx_lookup[nobj])
	
	cpdef rm_node_(DiGraph self,int nidx):
		"""
		Remove the node with index nidx.
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
		if self.first_free_node == -1:
			self.first_free_node = nidx
			self.node_info[nidx].indegree = -1
		else:
			self.node_info[nidx].indegree = self.first_free_node
			self.first_free_node = nidx
					
		self.num_nodes -= 1
	
	cpdef degree(DiGraph self,nobj):
		"""
		Return the degree of node with object nobj.
		"""
		return self.degree_(self.node_idx_lookup[nobj])

	cpdef degree_(DiGraph self,int nidx):
		"""
		Return the degree of node with index nidx.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node index %d' % nidx
			
		return self.node_info[nidx].indegree + self.node_info[nidx].outdegree
	
	cpdef in_degree(DiGraph self,nobj):
		"""
		Return the in-degree of node with object nobj.
		"""
		return self.in_degree_(self.node_idx_lookup[nobj])
	
	cpdef in_degree_(DiGraph self,int nidx):
		"""
		Return the in-degree of node with index nidx.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node index %d' % nidx
			
		return self.node_info[nidx].indegree
	
	cpdef out_degree(DiGraph self,nobj):
		"""
		Return the out-degree of node with object nobj.
		"""
		return self.out_degree_(self.node_idx_lookup[nobj])
	
	cpdef out_degree_(DiGraph self,int nidx):
		"""
		Return the out-degree of node with index nidx.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node index %d' % nidx
			
		return self.node_info[nidx].outdegree
	
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
	
	cpdef int add_edge(self, src, tgt, data=None, double weight=1) except -1:
		"""
		Add an edge to the graph from the src node to the tgt node.
		src and tgt are node objects.  If data is not None, then it
		is used as the data associated with this edge.
		
		This function returns the index for the new edge.
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
		Add an edge to the graph from the src node to the tgt node.
		src and tgt are node indices.  If data is not None, then it
		is used as the data associated with this edge.
	
		This function returns the index for the new edge.
		"""	
		cdef int i
		self.num_changes += 1
		
		if src >= self.node_capacity or not self.node_info[src].exists or tgt >= self.node_capacity or not self.node_info[tgt].exists:
			raise ZenException, 'Both source and destination nodes must exist (%d,%d)' % (src,tgt)
			
		cdef int eidx
		
		# recycle an edge index if possible
		if self.first_free_edge != -1:
			eidx = self.first_free_edge
			self.first_free_edge = self.edge_info[eidx].src
		else:
			eidx = self.next_edge_idx
			self.next_edge_idx += 1
		
		self.add_edge_x(eidx,src,tgt,data,weight)
		
		return eidx
		
	cpdef int add_edge_x(DiGraph self, int eidx, int src, int tgt, data, double weight) except -1:
		
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
		
		# source
		self.__insert_edge_into_outelist(src,eidx,tgt)
		
		# destination
		self.__insert_edge_into_inelist(tgt,eidx,src)	
		
		### Add edge info
		self.edge_info[eidx].exists = True
		self.edge_info[eidx].src = src
		self.edge_info[eidx].tgt = tgt
		self.edge_info[eidx].weight = weight
		
		#####
		# Done
		self.num_edges += 1
		
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
		Remove the edge between node objects src and tgt.
		"""
		self.rm_edge_(self.edge_idx(src,tgt))
	
	cpdef rm_edge_(DiGraph self,int eidx):
		"""
		Remove the edge with index eidx.
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
		if self.first_free_edge == -1:
			self.first_free_edge = eidx
			self.edge_info[eidx].src = -1
		else:
			self.edge_info[eidx].src = self.first_free_edge
			self.first_free_edge = eidx
				
		self.num_edges -= 1
		
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
		Return the node objects at the endpoints of the edge with index eidx.
		"""
		if not self.edge_info[eidx].exists:
			raise ZenException, 'Edge with ID %d does not exist' % eidx
	
		return self.node_obj_lookup[self.edge_info[eidx].src], self.node_obj_lookup[self.edge_info[eidx].tgt]
		
	cpdef endpoints_(DiGraph self,int eidx):
		"""
		Return the node indices at the endpoints of the edge with index eidx.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge index %d' % eidx

		return self.edge_info[eidx].src, self.edge_info[eidx].tgt

	cpdef endpoint(DiGraph self,int eidx,u):
		"""
		Return the other node (not u) that is the endpoint of this edge.  Note, no check is done
		to ensure that u is an endpoint of the edge.
		"""
		if u not in self.node_idx_lookup:
			raise ZenException, 'Invalid node object %s' % str(u)
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx

		return self.node_object(self.endpoint_(eidx,self.node_idx_lookup[u]))

	cpdef int endpoint_(DiGraph self,int eidx,int u) except -1:
		"""
		Return the other endpoint for edge eidx besides the one given (u).

		Note that this method is implemented for speed and no check is made to ensure that
		u is one of the edge's endpoints.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx

		if u == self.edge_info[eidx].src:
			return self.edge_info[eidx].tgt
		else:
			return self.edge_info[eidx].src

	cpdef src(DiGraph self,int eidx):
		return self.node_object(self.src_(eidx))

	cpdef int src_(DiGraph self,int eidx) except -1:
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx

		return self.edge_info[eidx].src
		
	cpdef tgt(DiGraph self,int eidx):
		return self.node_object(self.tgt_(eidx))
		
	cpdef int tgt_(DiGraph self,int eidx) except -1:
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge idx %d' % eidx

		return self.edge_info[eidx].tgt
		
	cpdef set_weight(DiGraph self,u,v,double w):
		self.set_weight_(self.edge_idx(u,v),w)

	cpdef set_weight_(DiGraph self,int eidx,double w):
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge index %d' % eidx
		
		self.edge_info[eidx].weight = w

	cpdef double weight(DiGraph self,u,v):
		return self.weight_(self.edge_idx(u,v))

	cpdef double weight_(DiGraph self,int eidx):
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge index %d' % eidx
			
		return self.edge_info[eidx].weight
		
	cpdef set_edge_data(DiGraph self,src,tgt,data):
		"""
		Associate a new data object with a specific edge in the network.
		If data is None, then any data associated with the edge is deleted.
		"""
		self.set_edge_data_(self.edge_idx(src,tgt),data)
		
	cpdef edge_data(DiGraph self,src,tgt):
		"""
		Return the data associated with the edge from src to tgt.
		"""
		return self.edge_data_(self.edge_idx(src,tgt))
	
	cpdef set_edge_data_(DiGraph self,int eidx,data):
		"""
		Associate a new data object with a specific edge in the network.
		If data is None, then any data associated with the edge is deleted.
		"""
		if eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge index %d' % eidx
			
		if data == None:
			if eidx in self.edge_data_lookup:
				del self.edge_data_lookup[eidx]
		else:
			self.edge_data_lookup[eidx] = data
	
	cpdef edge_data_(DiGraph self,int eidx,int dest=-1):
		"""
		Return the data associated with the edge with index eidx.
		
		If dest is specified, then the data associated with the edge
		connected nodes with indices eidx and dest is returned.
		"""			
		if dest != -1:
			eidx = self.edge_idx_(eidx,dest)			
		elif eidx >= self.edge_capacity or not self.edge_info[eidx].exists:
			raise ZenException, 'Invalid edge index %d' % eidx
	
		if eidx in self.edge_data_lookup:
			return self.edge_data_lookup[eidx]
		else:
			return None
			
	cpdef bool has_edge(DiGraph self,src,tgt):
		"""
		Return True if the graph contains an edge connecting node objects src and tgt.
		
		If src or tgt are not in the graph, then this function returns False.
		"""
		if src not in self.node_idx_lookup:
			return False

		if tgt not in self.node_idx_lookup:
			return False
		
		src = self.node_idx_lookup[src]
		tgt = self.node_idx_lookup[tgt]
		
		return self.has_edge_(src,tgt)
		
	cpdef bool has_edge_(DiGraph self,int src,int tgt):
		"""
		Return True if the graph contains an edge connecting nodes with indices src and tgt.
		
		Both src and tgt must be valid indices.
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
		Return the edge index for the edge connecting node objects src and tgt.
		"""
		src = self.node_idx_lookup[src]
		tgt = self.node_idx_lookup[tgt]
		return self.edge_idx_(src,tgt,data)
	
	cpdef edge_idx_(DiGraph self, int src, int tgt, data=False):
		"""
		Return the edge index for the edge connecting node indices src and tgt.
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
		
		If nobj is None, then all edges in the graph are iterated over.  Otherwise
		the edges touching the node with object nobj are iterated over.
		
		If data is False, then the iterator returns a tuple (src,tgt).  Otherwise, the
		iterator returns a tuple (src,tgt,data).
		"""
		if nobj is None:
			return AllEdgeIterator(self,weight,data,True)
		else:
			return NodeEdgeIterator(self,self.node_idx_lookup[nobj],ITER_BOTH,weight,data,True)
	
	cpdef edges_iter_(DiGraph self,int nidx=-1,data=False,weight=False):
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
				
			return NodeEdgeIterator(self,nidx,ITER_BOTH,weight,data,False)
	
	cpdef edges(DiGraph self,nobj=None,data=False,weight=False):
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
			idx = 0
			num_edges = self.node_info[nidx].indegree
			elist = self.node_info[nidx].inelist
			for i in range(num_edges):
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
			
			# loop over out edges - this is copied verbatim from out_edges_iter(...)			
			num_edges = self.node_info[nidx].outdegree
			elist = self.node_info[nidx].outelist
			for i in range(num_edges):
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
				
	cpdef edges_(DiGraph self,int nidx=-1,data=False,weight=False):
		"""
		Return a numpy array of edges.  If nidx is None, then all edges will be returned.
		If nidx is not None, then the edges for the node with nidx will be returned.
		
		If data is True, then the numpy array will contain an additional column containing the
		data for each edge.
		
		If weight is True, then the numpy array will contain an additional column containing the weight
		of each edge.
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
		Return an iterator over the inbound edges of node with object nobj.  If data is 
		True then tuples (src,tgt,data) are returned.
		"""
		return NodeEdgeIterator(self,self.node_idx_lookup[nobj],ITER_INDEGREE,weight,data,True)

	cpdef in_edges_iter_(DiGraph self,int nidx,data=False,weight=False):
		"""
		Return an iterator over the inbound edges of node with index nidx.  If data is
		True then tuples (edge idx,data) are returned.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx
			
		return NodeEdgeIterator(self,nidx,ITER_INDEGREE,weight,data)
		
	cpdef in_edges(DiGraph self,nobj,data=False,weight=False):
		"""
		Return a list containing the inbound edges of the node with object nobj.  If
		data is True, then the list contains tuples (src,tgt,data).
		"""
		return list(self.in_edges_iter(nobj,weight,data))

	cpdef in_edges_(DiGraph self,int nidx,data=False,weight=False):
		"""
		Return a numpy array containing the inbound edges of the node with index nidx.
		If data is False, the result is 1-D array containing edge indices.  If data
		is True, the result is a 2-D array containing the edge index and data.
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
				result[i,1] = self.edge_data_lookup[elist[i]]
				result[i,2] = self.edge_info[elist[i]].weight
		elif data:
			result = numpy.empty( (indegree,2), dtype=numpy.object_)
			
			for i in range(indegree):
				result[i,0] = elist[i]
				result[i,1] = self.edge_data_lookup[elist[i]]
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
		Return an iterator over the outbound edges of node with object nobj.  If data is 
		True then tuples (src,tgt,data) are returned.
		"""
		return NodeEdgeIterator(self,self.node_idx_lookup[nobj],ITER_OUTDEGREE,weight,data,True)

	cpdef out_edges_iter_(DiGraph self,int nidx,data=False,weight=False):
		"""
		Return an iterator over the outbound edges of node with index nidx.  If data is
		True then tuples (edge idx,data) are returned.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx
			
		return NodeEdgeIterator(self,nidx,ITER_OUTDEGREE,weight,data)
	
	cpdef out_edges(DiGraph self,nobj,data=False,weight=False):
		"""
		Return a list containing the outbound edges of the node with object nobj.  If
		data is True, then the list contains tuples (src,tgt,data).
		"""
		return list(self.out_edges_iter(nobj,weight,data))

	cpdef out_edges_(DiGraph self,int nidx,data=False,weight=False):
		"""
		Return a numpy array containing the outbound edges of the node with index nidx.
		If data is False, the result is 1-D array containing edge indices.  If data
		is True, the result is a 2-D array containing the edge index and data.
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
				result[i,1] = self.edge_data_lookup[elist[i]]
				result[i,2] = self.edge_info[elist[i]].weight
		elif data:
			result = numpy.empty( (outdegree,2), dtype=numpy.object_)
			
			for i in range(outdegree):
				result[i,0] = elist[i]
				result[i,1] = self.edge_data_lookup[elist[i]]
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
		Return an iterator over the edges of nodes in nbunch.  If data is 
		True then tuples (src,tgt,data) are returned.
		"""
		return SomeEdgeIterator(self,[self.node_idx_lookup[x] for x in nbunch],ITER_BOTH,weight,data,True)

	cpdef grp_in_edges_iter(DiGraph self,nbunch,data=False,weight=False):
		"""
		Return an iterator over the inbound edges of nodes in nbunch.  If data is 
		True then tuples (src,tgt,data) are returned.
		"""
		return SomeEdgeIterator(self,[self.node_idx_lookup[x] for x in nbunch],ITER_INDEGREE,weight,data,True)

	cpdef grp_out_edges_iter(DiGraph self,nbunch,data=False,weight=False):
		"""
		Return an iterator over the outbound edges of nodes in nbunch.  If data is 
		True then tuples (src,tgt,data) are returned.
		"""
		return SomeEdgeIterator(self,[self.node_idx_lookup[x] for x in nbunch],ITER_OUTDEGREE,weight,data,True)
		
	cpdef grp_edges_iter_(DiGraph self,nbunch,data=False,weight=False):
		"""
		Return an iterator over the edges of node indices in nbunch.  If data is 
		True then tuples (eidx,data) are returned.
		"""
		for nidx in nbunch:
			if nidx >= self.node_capacity or not self.node_info[nidx].exists:
				raise ZenException, 'Invalid node idx %d' % nidx
				
		return SomeEdgeIterator(self,nbunch,ITER_BOTH,weight,data)

	cpdef grp_in_edges_iter_(DiGraph self,nbunch,data=False,weight=False):
		"""
		Return an iterator over the inbound edges of node indices in nbunch.  If data is 
		True then tuples (eidx,data) are returned.
		"""
		for nidx in nbunch:
			if nidx >= self.node_capacity or not self.node_info[nidx].exists:
				raise ZenException, 'Invalid node idx %d' % nidx
				
		return SomeEdgeIterator(self,nbunch,ITER_INDEGREE,weight,data)

	cpdef grp_out_edges_iter_(DiGraph self,nbunch,data=False,weight=False):
		"""
		Return an iterator over the outbound edges of node indices in nbunch.  If data is 
		True then tuples (eidx,data) are returned.
		"""
		for nidx in nbunch:
			if nidx >= self.node_capacity or not self.node_info[nidx].exists:
				raise ZenException, 'Invalid node idx %d' % nidx
				
		return SomeEdgeIterator(self,nbunch,ITER_OUTDEGREE,weight,data)
						
	cpdef neighbors(DiGraph self,nobj,data=False):
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
		Return an iterator over the neighbors of node with object nobj.  If data is 
		True then tuples (obj,data) are returned.
		"""
		return NeighborIterator(self,self.node_idx_lookup[nobj],ITER_BOTH,False,data,True)
		
	cpdef neighbors_iter_(DiGraph self,int nidx,obj=False,data=False):
		"""
		Return an iterator over the neighbors of node with index nidx.  If obj is True, then
		tuples (nidx,obj) are returned.  If data is True then tuples (obj,data) are returned.
		If both are True, then tuples (nix,obj,data) are returned.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx
			
		return NeighborIterator(self,nidx,ITER_BOTH,obj,data,False)
		
	cpdef in_neighbors_iter(DiGraph self,nobj,data=False):
		"""
		Return an iterator over the in-neighbors of node with object nobj.  If data is 
		True then tuples (obj,data) are returned.
		"""
		return NeighborIterator(self,self.node_idx_lookup[nobj],ITER_INDEGREE,False,data,True)
		
	cpdef in_neighbors_iter_(DiGraph self,int nidx,obj=False,data=False):
		"""
		Return an iterator over the in-neighbors of node with index nidx.  If obj is True, then
		tuples (nidx,obj) are returned.  If data is True then tuples (obj,data) are returned.
		If both are True, then tuples (nix,obj,data) are returned.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx
			
		return NeighborIterator(self,nidx,ITER_INDEGREE,obj,data,False)

	cpdef in_neighbors(DiGraph self,nobj,data=False):
		"""
		Return a list of nodes that are in-neighbors of the node nobj.  If data is True, then a 
		list of tuples is returned, each tuple containing a neighbor node object and its data.
		"""
		return list(self.in_neighbors_iter(nobj,data))

	cpdef in_neighbors_(DiGraph self,int nidx,obj=False,data=False):
		"""
		Return a numpy array of node ids corresponding to in-neighbors of the node with index nidx.
		
		If obj and data are False, then the numpy array will be a 1-D array containing node indices.  If obj or data
		are True, then the numpy array will be a 2-D array containing indices in the first column and
		the node object/data object in the second column.  If both obj and data are True, then the numpy array will be
		a 2-D array containing indices in the first column, the node object in the second, and the node data in 
		the third column.
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
		Return an iterator over the out-neighbors of node with object nobj.  If data is 
		True then tuples (obj,data) are returned.
		"""
		return NeighborIterator(self,self.node_idx_lookup[nobj],ITER_OUTDEGREE,False,data,True)
		
	cpdef out_neighbors_iter_(DiGraph self,int nidx,obj=False,data=False):
		"""
		Return an iterator over the in-neighbors of node with index nidx.  If obj is True, then
		tuples (nidx,obj) are returned.  If data is True then tuples (obj,data) are returned.
		If both are True, then tuples (nix,obj,data) are returned.
		"""
		if nidx >= self.node_capacity or not self.node_info[nidx].exists:
			raise ZenException, 'Invalid node idx %d' % nidx
			
		return NeighborIterator(self,nidx,ITER_OUTDEGREE,obj,data,False)

	cpdef out_neighbors(DiGraph self,nobj,data=False):
		"""
		Return a list of nodes that are out-neighbors of the node nobj.  If data is True, then a 
		list of tuples is returned, each tuple containing a neighbor node object and its data.
		"""
		return list(self.out_neighbors_iter(nobj,data))

	cpdef out_neighbors_(DiGraph self,int nidx,obj=False,data=False):
		"""
		Return a numpy array of node ids corresponding to out-neighbors of the node with index nidx.
		
		If obj and data are False, then the numpy array will be a 1-D array containing node indices.  If obj or data
		are True, then the numpy array will be a 2-D array containing indices in the first column and
		the node object/data object in the second column.  If both obj and data are True, then the numpy array will be
		a 2-D array containing indices in the first column, the node object in the second, and the node data in 
		the third column.
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
		Return an iterator over the neighbors of nodes in nbunch.  If data is 
		True then tuples (nobj,data) are returned.
		"""
		return SomeNeighborIterator(self,[self.node_idx_lookup[x] for x in nbunch],ITER_BOTH,False,data,True)

	cpdef grp_in_neighbors_iter(DiGraph self,nbunch,data=False):
		"""
		Return an iterator over the in-neighbors of nodes in nbunch.  If data is 
		True then tuples (nobj,data) are returned.
		"""
		return SomeNeighborIterator(self,[self.node_idx_lookup[x] for x in nbunch],ITER_INDEGREE,False,data,True)

	cpdef grp_out_neighbors_iter(DiGraph self,nbunch,data=False):
		"""
		Return an iterator over the out-neighbors of nodes in nbunch.  If data is 
		True then tuples (nobj,data) are returned.
		"""
		return SomeNeighborIterator(self,[self.node_idx_lookup[x] for x in nbunch],ITER_OUTDEGREE,False,data,True)

	cpdef grp_neighbors_iter_(DiGraph self,nbunch,obj=False,data=False):
		"""
		Return an iterator over the neighbors of nodes in nbunch.  If obj is True
		then tuples (nidx,nobj) are returned.  If data is True then tuples (nidx,data) 
		are returned.  If both are True, then tuples (nidx,nobj,data) are returned.
		"""
		for nidx in nbunch:
			if nidx >= self.node_capacity or not self.node_info[nidx].exists:
				raise ZenException, 'Invalid node idx %d' % nidx
				
		return SomeNeighborIterator(self,nbunch,ITER_BOTH,obj,data,False)

	cpdef grp_in_neighbors_iter_(DiGraph self,nbunch,obj=False,data=False):
		"""
		Return an iterator over the in-neighbors of nodes in nbunch.  If obj is True
		then tuples (nidx,nobj) are returned.  If data is True then tuples (nidx,data) 
		are returned.  If both are True, then tuples (nidx,nobj,data) are returned.
		"""
		for nidx in nbunch:
			if nidx >= self.node_capacity or not self.node_info[nidx].exists:
				raise ZenException, 'Invalid node idx %d' % nidx
				
		return SomeNeighborIterator(self,nbunch,ITER_INDEGREE,obj,data,False)

	cpdef grp_out_neighbors_iter_(DiGraph self,nbunch,obj=False,data=False):
		"""
		Return an iterator over the out-neighbors of nodes in nbunch.  If obj is True
		then tuples (nidx,nobj) are returned.  If data is True then tuples (nidx,data) 
		are returned.  If both are True, then tuples (nidx,nobj,data) are returned.
		"""
		for nidx in nbunch:
			if nidx >= self.node_capacity or not self.node_info[nidx].exists:
				raise ZenException, 'Invalid node idx %d' % nidx
				
		return SomeNeighborIterator(self,nbunch,ITER_OUTDEGREE,obj,data,False)
								
cdef class NodeIterator:
	cdef bool data
	cdef DiGraph graph
	cdef int idx
	cdef int node_count
	cdef bool nobj
	cdef bool obj
	cdef long init_num_changes
	
	def __cinit__(NodeIterator self,DiGraph graph,bool obj,bool data,bool nobj):
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
	cdef bool data
	cdef bool weight
	cdef DiGraph graph
	cdef int idx
	cdef bool endpoints
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
	cdef bool data
	cdef bool weight
	cdef DiGraph graph
	cdef int nidx
	cdef int deg
	cdef int idx
	cdef int which_degree
	cdef bool endpoints
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
	cdef bool data
	cdef bool weight
	cdef DiGraph graph
	cdef int which_degree
	cdef touched_edges
	cdef nbunch_iter
	cdef edge_iter
	cdef bool endpoints
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
	cdef bool data
	cdef deg
	cdef int nidx
	cdef DiGraph G
	cdef touched_nodes
	cdef bool use_nobjs
	cdef bool obj
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
	cdef bool data
	cdef DiGraph graph
	cdef int idx
	cdef int which_degree
	cdef touched_nodes
	cdef nbunch_iter
	cdef neighbor_iter
	cdef bool use_nobjs
	cdef bool obj
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