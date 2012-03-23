#cython: embedsignature=True

import numpy
import ctypes
import types
from graph cimport Graph

cimport libc.stdlib as stdlib

# some things not defined in the cython stdlib header
cdef extern from "stdlib.h" nogil:
	void* memmove(void* destination, void* source, size_t size)

cdef extern from "math.h":
	double ceil(double x)
	double floor(double x)

__all__ = ['HyperGraph']

cdef struct NodeInfo:
	bint exists
	
	int degree
	int* elist
	int capacity
	
cdef int sizeof_NodeInfo = sizeof(NodeInfo)

cdef struct EdgeInfo:
	bint exists
	int* nlist
	int num_nodes
	int capacity
	
cdef int sizeof_EdgeInfo = sizeof(EdgeInfo)

cdef class HyperGraph:
	
	def __cinit__(HyperGraph self,node_capacity=100,edge_capacity=100,edge_list_capacity=5):
		self.node_grow_factor = 1.5
		self.edge_list_grow_factor = 1.5
		self.edge_grow_factor = 1.5
		
		self.edge_list_capacity = edge_list_capacity
		
		self.num_nodes = 0
		self.node_capacity = node_capacity
		self.next_node_idx = 0
		self.node_info = <NodeInfo*> stdlib.malloc(sizeof_NodeInfo*self.node_capacity)
		self.node_obj_lookup = {}
		self.node_data_lookup = {}
		self.node_idx_lookup = {}
		
		self.num_edges = 0
		self.edge_capacity = edge_capacity
		self.next_edge_idx = 0
		self.edge_info = <EdgeInfo*> stdlib.malloc(sizeof_EdgeInfo*self.edge_capacity)
		self.edge_idx_lookup = {}
		self.edge_data_lookup = {}
		
	def __dealloc__(HyperGraph self):
		cdef int i
		
		# deallocate all node data (node_info)
		for i in range(self.next_node_idx):
			if self.node_info[i].exists:
				stdlib.free(self.node_info[i].elist)
		stdlib.free(self.node_info)
		
		# deallocate all edge data (edge_info)
		for i in range(self.next_edge_idx):
			if self.edge_info[i].exists:
				stdlib.free(self.edge_info[i].nlist)
		stdlib.free(self.edge_info)

	cpdef Graph to_graph(HyperGraph self):
		cdef Graph G = Graph()
		cdef int i,j
		
		# add all nodes
		for nobj,ndata in self.nodes_iter(data=True):
			G.add_node(nobj,ndata)
		
		# add all edges
		for epts,edata in self.edges_iter(data=True):
			for i in range(len(epts)):
				for j in range(i+1,len(epts)):
					u = epts[i]
					v = epts[j]
					if not G.has_edge(u,v):
						G.add_edge(u,v,edata)
						
		return G
			
	cpdef bool is_directed(HyperGraph self):
		return False

	cpdef int add_node(HyperGraph self,nobj=None,data=None):
		cdef int node_idx = self.next_node_idx
		self.next_node_idx += 1

		if nobj is not None:
			self.node_idx_lookup[nobj] = node_idx
			self.node_obj_lookup[node_idx] = nobj

		if data is not None:
			self.node_data_lookup[node_idx] = data

		# grow the node_info array as necessary
		cdef int new_node_capacity
		if node_idx >= self.node_capacity:
			new_node_capacity = <int> ceil( float(self.node_capacity) * self.node_grow_factor)
			self.node_info = <NodeInfo*> stdlib.realloc(self.node_info, sizeof_NodeInfo*new_node_capacity)
			self.node_capacity = new_node_capacity

		self.node_info[node_idx].exists = True
		self.node_info[node_idx].degree = 0

		# initialize edge lists
		self.node_info[node_idx].elist = <int*> stdlib.malloc(sizeof(int) * self.edge_list_capacity)
		self.node_info[node_idx].capacity = self.edge_list_capacity

		self.num_nodes += 1
		return node_idx

	def __contains__(HyperGraph self,nobj):
		"""
		Return True if the node object is in the graph.
		"""
		return nobj in self.node_idx_lookup

	cpdef int node_idx(HyperGraph self,nobj):
		return self.node_idx_lookup[nobj]

	cpdef node_object(HyperGraph self,int nidx):
		if nidx in self.node_obj_lookup:
			return self.node_obj_lookup[nidx]
		else:
			return None

	cpdef node_data(HyperGraph self,nobj):
		return self.node_data_(self.node_idx_lookup[nobj])

	cpdef node_data_(HyperGraph self,int nidx):
		if nidx in self.node_data_lookup:
			return self.node_data_lookup[nidx]
		else:
			return None

	cpdef nodes_iter(HyperGraph self,data=False):
		return NodeIterator(self,data,True)

	cpdef nodes_iter_(HyperGraph self,data=False):
		return NodeIterator(self,data,False)

	cpdef nodes(HyperGraph self,data=False):
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
					raise Exception, 'Node (idx=%d) is missing object' % i
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

	cpdef nodes_(HyperGraph self,data=False):
		"""
		Return a numpy array of the nodes.  If data is False, then the result is a
		1-D array of the node indices.  If data is True, then the result is a 2-D 
		array in which the first column is node indices and the second column are
		node descriptors.
		"""
		ndim = 1 if not data else 2

		result = numpy.empty( (self.num_nodes,ndim), dtype=numpy.object_)

		cdef int idx = 0
		cdef int i = 0
		while idx < self.num_nodes:
			if self.node_info[i].exists:
				if data:
					result[idx,0] = i
					if i in self.node_data_lookup:
						result[idx,1] = self.node_data_lookup[i]
					else:
						result[idx,1] = None
				else:
					result[idx] = i
				idx += 1

			i += 1

		return result

	cpdef rm_node(HyperGraph self,nobj):
		self.rm_node_(self.node_idx_lookup[nobj])

	cpdef rm_node_(HyperGraph self,int nidx):
		cdef int i

		# remove all edges
		for i in range(self.node_info[nidx].degree-1,-1,-1):
			eidx = self.node_info[nidx].elist[i]

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

		self.num_nodes -= 1

	cpdef degree(HyperGraph self,nobj):
		return self.degree_(self.node_idx_lookup[nobj])

	cpdef degree_(HyperGraph self,int nidx):
		return self.node_info[nidx].degree

	def __getitem__(HyperGraph self,nobj):
		"""
		Get the descriptor for the node with the object given
		"""
		return self.node_data_lookup[self.node_idx_lookup[nobj]]

	def __len__(HyperGraph self):
		return self.num_nodes

	def size(HyperGraph self):
		return self.num_edges

	def compact(self):
		raise Exception, 'NOT IMPLEMENTED'

	cpdef add_edge(HyperGraph self, node_list, data=None):
		cdef int nidx1, nidx2

		for n in node_list:
			if n not in self.node_idx_lookup:
				self.add_node(n)

		return self.add_edge_([self.node_idx_lookup[n] for n in node_list],data)

	cpdef int add_edge_(HyperGraph self, node_list, data=None):
		"""
		Add an edge that connects all nodes in the node_list.  node_list
		contains only node indices.
		"""
		cdef int nidx
		
		# sanity check
		for nidx in node_list:
			if not self.node_info[nidx].exists:
				raise Exception, 'Node %d does not exist in the graph' % nidx
		
		node_list.sort()
				
		cdef int eidx = self.next_edge_idx
		self.next_edge_idx += 1

		if data is not None:
			self.edge_data_lookup[eidx] = data

		# grow the info array
		if eidx >= self.edge_capacity:
			self.edge_capacity = <int> ceil( <float>self.edge_capacity * self.edge_grow_factor)
			self.edge_info = <EdgeInfo*> stdlib.realloc( self.edge_info, sizeof_EdgeInfo * self.edge_capacity)

		self.edge_info[eidx].exists = True
		self.edge_info[eidx].nlist = <int*> stdlib.malloc( len(node_list) * sizeof(int))
		self.edge_info[eidx].capacity = len(node_list)
		self.edge_info[eidx].num_nodes = len(node_list)
		
		# store away connectivity
		cdef int new_capacity
		cdef int num_edges
		cdef int pos
		cdef double elist_len
		for i,nidx in enumerate(node_list):
			
			# store this node in the edge structure
			self.edge_info[eidx].nlist[i] = nidx
		
			####
			# connect up the edges to nodes
			num_edges = self.node_info[nidx].degree
			elist_len = <double> self.node_info[nidx].capacity
			if num_edges >= elist_len:
				new_capacity = <int>ceil(elist_len * self.edge_list_grow_factor)
				self.node_info[nidx].elist = <int*> stdlib.realloc(self.node_info[nidx].elist, sizeof(int) * new_capacity)
				self.node_info[nidx].capacity = new_capacity
			self.node_info[nidx].elist[num_edges] = eidx
			self.node_info[nidx].degree += 1

		#####
		# Done
		self.num_edges += 1
		return eidx

	cpdef rm_edge(HyperGraph self,src,tgt):
		self.rm_edge_(self.edge_idx(self.edge_idx_lookup[src],self.edge_idx_lookup[tgt]))

	cpdef rm_edge_(HyperGraph self,int eidx):
		"""
		Comments here.
		"""

		cdef int i,j
		cdef int nidx
		#####
		# remove entries in nodes
		for i in range(self.edge_info[eidx].num_nodes):
			nidx = self.edge_info[eidx].nlist[i]
			
			# find the edge idx
			j = 0
			while j < self.node_info[nidx].degree and self.node_info[nidx].elist[j] != eidx:
				j += 1

			self.node_info[nidx].elist[i] = self.node_info[nidx].elist[self.node_info[nidx].degree-1]
			self.node_info[nidx].degree -= 1

		# remove actual data structure
		self.edge_info[eidx].exists = False
		stdlib.free(self.edge_info[eidx].nlist)
		
		if eidx in self.edge_data_lookup:
			data = self.edge_data_lookup[eidx]
			del self.edge_data_lookup[eidx]

		self.num_edges -= 1

		return

	cpdef card(HyperGraph self,nlist):
		"""
		Return the cardinality of an edge.
		"""
		return self.card([self.node_idx_lookup[x] for x in nlist])
		
	cpdef card_(HyperGraph self,nlist):
		"""
		Return the cardinality of an edge.  If nlist is an integer, it's interpreted
		as an edge idx.  Otherwise, it's taken to be a node list that identifies
		the edge.
		"""
		if type(nlist) == types.IntType:
			return self.edge_info[nlist].num_nodes
		else:
			return self.edge_info[self.edge_idx_(nlist)].num_nodes

	cpdef endpoints(HyperGraph self,int eidx):
		cdef int nidx
		
		if not self.edge_info[eidx].exists:
			raise Exception, 'Edge with ID %d does not exist' % eidx
		
		eps = []
		for i in range(self.edge_info[eidx].num_nodes):
			nidx = self.edge_info[eidx].nlist[i]
			eps.append(self.node_obj_lookup[nidx])
			
		return eps

	cpdef endpoints_(HyperGraph self,int eidx):
		cdef int nidx
	
		if not self.edge_info[eidx].exists:
			raise Exception, 'Edge with ID %d does not exist' % eidx
	
		eps = []
		for i in range(self.edge_info[eidx].num_nodes):
			nidx = self.edge_info[eidx].nlist[i]
			eps.append(nidx)
		
		return eps

	cpdef edge_data(HyperGraph self,nlist):
		return self.edge_data_(self.edge_idx(nlist))

	cpdef edge_data_(HyperGraph self,nlist):
		cdef int eidx
		if type(nlist) == types.IntType:
			eidx = nlist
			if not self.edge_info[eidx].exists:
				raise Exception, 'Edge with ID %d does not exist' % eidx
		else:
			eidx = self.edge_idx_(nlist)

		if eidx in self.edge_data_lookup:
			return self.edge_data_lookup[eidx]
		else:
			return None

	cpdef has_edge(HyperGraph self,nlist):
		return self.has_edge_([self.node_idx_lookup[x] for x in nlist])

	cpdef bool has_edge_(HyperGraph self,nlist):
		nlist.sort()
		cdef int n0 = nlist[0]
		cdef int i, j, eidx
		
		for i in range(self.node_info[n0].degree):
			eidx = self.node_info[n0].elist[i]
			
			# check for an exact match
			if self.edge_info[eidx].num_nodes != len(nlist):
				continue
			
			match = True
			for j in range(self.edge_info[eidx].num_nodes):
				if self.edge_info[eidx].nlist[j] != nlist[j]:
					match = False
					break
					
			if match:
				return True

	cpdef edge_idx(HyperGraph self, nlist, data=False):
		return self.edge_idx_([self.node_idx_lookup[x] for x in nlist],data)

	cpdef edge_idx_(HyperGraph self, nlist, data=False):
		nlist.sort()
		cdef int n0 = nlist[0]
		cdef int i, j, eidx
	
		for i in range(self.node_info[n0].degree):
			eidx = self.node_info[n0].elist[i]
		
			# check for an exact match
			if self.edge_info[eidx].num_nodes != len(nlist):
				continue
		
			match = True
			for j in range(self.edge_info[eidx].num_nodes):
				if self.edge_info[eidx].nlist[j] != nlist[j]:
					match = False
					break
				
			if match:
				if data:
					return eidx, self.edge_data_lookup[eidx]
				else:
					return eidx		
					
		raise Exception, 'Edge %s does not exist.' % str(nlist)

	cpdef edges_iter(HyperGraph self,nobj=None,data=False):		
		if nobj is None:
			return AllEdgeIterator(self,data,True)
		else:
			return NodeEdgeIterator(self.node_idx_lookup[nobj],data,True)

	cpdef edges_iter_(HyperGraph self,int nidx=-1,data=False):
		if nidx == -1:
			return AllEdgeIterator(self,data,False)
		else:
			return NodeEdgeIterator(self,nidx,data,False)

	cpdef edges(HyperGraph self,nobj=None,data=False):
		"""
		Return edges connected to a node.  If nobj is not specified, then
		all edges in the network are returned.
		"""
		cdef int num_edges
		cdef int* elist
		cdef int i
		cdef nidx = -1

		if nobj is not None:
			nidx = self.node_idx_lookup[nobj]

		# iterate over all edges
		cdef int* nlist
		result = []
		if nidx == -1:
			idx = 0
			for i in range(self.next_edge_idx):
				if self.edge_info[i].exists:
					nlist = self.edge_info[i].nlist
					if data is True:
						result.append( ([self.node_obj_lookup[nlist[i]] for i in range(self.edge_info[i].num_nodes)], self.edge_data_lookup[i]) )
					else:
						result.append( [self.node_obj_lookup[nlist[i]] for i in range(self.edge_info[i].num_nodes)] )
					idx += 1

			return result
		else:
			idx = 0
			num_edges = self.node_info[nidx].degree
			elist = self.node_info[nidx].elist
			for i in range(num_edges):
				nlist = self.edge_info[elist[i]].nlist
				if data is True:
					result.append( ([self.node_obj_lookup[nlist[i]] for i in range(self.edge_info[elist[i]].num_nodes)],self.edge_data_lookup[elist[i]]) )
				else:
					result.append( [self.node_obj_lookup[nlist[i]] for i in range(self.edge_info[elist[i]].num_nodes)] )
				idx += 1

			return result

	cpdef edges_(HyperGraph self,int nidx=-1,data=False):
		"""
		Return a numpy array of edges.  If nbunch is None, then all edges will be returned.
		If nbunch is not None, then the edges for all nodes in nbunch will be returned.

		If data is False, then the numpy array will be a 1-D array containing edge indices.  If data
		is True, then the numpy array will be a 2-D array containing indices in the first column and
		descriptor objects in the second column.
		"""
		cdef int num_edges
		cdef int* elist
		cdef int* nlist
		cdef int i

		# iterate over all edges
		result = None
		if nidx == -1:
			if data:
				result = numpy.empty( (self.num_edges, 2), dtype=numpy.object_)
			else:
				result = numpy.empty(self.num_edges, dtype=numpy.object_)

			idx = 0
			for i in range(self.next_edge_idx):
				if self.edge_info[i].exists:
					if data is True:
						result[idx,0] = i
						result[idx,1] = self.edge_data_lookup[i]
					else:
						result[idx] = i
					idx += 1

			return result
		else:
			if data:
				result = numpy.empty( (self.node_info[nidx].degree,2), dtype=numpy.object_)
			else:
				result = numpy.empty(self.node_info[nidx].degree, dtype=numpy.object_)

			idx = 0
			num_edges = self.node_info[nidx].degree
			elist = self.node_info[nidx].elist
			for i in range(num_edges):
				if data is True:
					result[idx,0] = elist[i]
					result[idx,1] = self.edge_data_lookup[elist[i]]
				else:
					result[idx] = elist[i]
				idx += 1

			return result

	cpdef grp_edges_iter(HyperGraph self,nbunch,data=False):
		return SomeEdgeIterator(self,[self.node_idx_lookup[x] for x in nbunch],data,True)

	cpdef grp_edges_iter_(HyperGraph self,nbunch,data=False):
		return SomeEdgeIterator(self,nbunch,data)
		
	cpdef is_neighbor(HyperGraph self,nobj1,nobj2):
		"""
		Return True if nobj1 and nobj2 are neighbors.
		"""
		return self.is_neighbor_(self.node_idx_lookup[nobj1],self.node_idx_lookup[nobj2])

	cpdef is_neighbor_(HyperGraph self, int nidx1, int nidx2):
		cdef int i,j
		cdef int eidx
		
		for i in range(self.node_info[nidx1].degree):
			eidx = self.node_info[nidx1].elist[i]
			
			for j in range(self.edge_info[eidx].num_nodes):
				if self.edge_info[eidx].nlist[j] == nidx2:
					return True
					
		return False

	cpdef neighbors(HyperGraph self,nobj,data=False):
		"""
		Return a list of nodes that are neighbors of the node nobj.  If data is True, then a 
		list of tuples is returned, each tuple containing a neighbor node object and its data.
		"""
		cdef int num_edges
		cdef int* elist
		cdef int* nlist
		cdef int num_nodes
		cdef int rid
		cdef int i,j
		cdef nidx = self.node_idx_lookup[nobj]

		visited_neighbors = set()

		result = []

		# loop over in edges
		num_edges = self.node_info[nidx].degree
		elist = self.node_info[nidx].elist
		for i in range(num_edges):
			nlist = self.edge_info[elist[i]].nlist
			num_nodes = self.edge_info[elist[i]].num_nodes
			
			for j in range(num_nodes):
				rid = nlist[j]
				if rid == nidx or rid in visited_neighbors:
					continue

				if data is True:
					result.append( (self.node_obj_lookup[rid], self.node_data_lookup[rid]) )
				else:
					if rid not in self.node_obj_lookup:
						raise Exception, 'No node lookup known for node %d' % rid
					result.append(self.node_obj_lookup[rid])
				visited_neighbors.add(rid)

		return result

	cpdef neighbors_(HyperGraph self,int nidx,data=False):
		"""
		Return a numpy array of node ids corresponding to all neighbors of the node with id nid.

		If data is False, then the numpy array will be a 1-D array containing node indices.  If data
		is True, then the numpy array will be a 2-D array containing indices in the first column and
		descriptor objects in the second column.
		"""
		cdef int num_edges
		cdef int* elist
		cdef int* nlist
		cdef int num_nodes
		cdef int rid

		visited_neighbors = set()

		result = None
		if data:
			result = numpy.empty( (self.node_info[nidx].degree,2), dtype=numpy.object_)
		else:
			result = numpy.empty( self.node_info[nidx].degree, dtype=numpy.object_)

		idx = 0

		# loop over in edges
		num_edges = self.node_info[nidx].degree
		elist = self.node_info[nidx].elist
		for i in range(num_edges):
			nlist = self.edge_info[elist[i]].nlist
			num_nodes = self.edge_info[elist[i]].num_nodes
			
			for j in range(num_nodes):
				rid = nlist[j]

			if rid == nidx or rid in visited_neighbors:
				continue

			visited_neighbors.add(rid)

			if data is True:
				result[idx,0] = rid
				result[idx,1] = self.node_data_lookup[rid]
			else:
				result[idx] = rid
			idx += 1

		return result

	cpdef neighbors_iter(HyperGraph self,nobj,data=False):
		return NeighborIterator(self,self.node_idx_lookup[nobj],data,True)

	cpdef neighbors_iter_(HyperGraph self,int nidx,data=False):
		return NeighborIterator(self,nidx,data,False)

	cpdef grp_neighbors_iter(HyperGraph self,nbunch,data=False):
		return SomeNeighborIterator(self,[self.node_idx_lookup[x] for x in nbunch],data,True)

	cpdef grp_neighbors_iter_(HyperGraph self,nbunch,data=False):
		return SomeNeighborIterator(self,nbunch,data,False)

cdef class NodeIterator:
	cdef bool data
	cdef HyperGraph graph
	cdef int idx
	cdef int node_count
	cdef bool nobj

	def __cinit__(NodeIterator self,HyperGraph graph,bool data,bool nobj):
		self.data = data
		self.graph = graph
		self.idx = 0
		self.node_count = 0
		self.nobj = nobj

	def __next__(NodeIterator self):
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

		if self.data:
			val = None
			if idx in self.graph.node_data_lookup:
				val = self.graph.node_data_lookup[idx]

			if self.nobj:
				if idx not in self.graph.node_obj_lookup:
					raise Exception, 'Node (idx=%d) missing object' % idx
				return self.graph.node_obj_lookup[idx], val
			else:
				return idx, val
		else:
			if self.nobj:
				if idx not in self.graph.node_obj_lookup:
					raise Exception, 'Node (idx=%d) missing object' % idx
				return self.graph.node_obj_lookup[idx]
			else:
				return idx

	def __iter__(NodeIterator self):
		return self

cdef class AllEdgeIterator:
	cdef bool data
	cdef HyperGraph graph
	cdef int idx
	cdef bool objs

	def __cinit__(AllEdgeIterator self,HyperGraph graph,data=False,objs=False):
		self.graph = graph
		self.data = data
		self.idx = 0
		self.objs = objs

	def __next__(AllEdgeIterator self):
		cdef int idx = self.idx
		cdef int i
		
		if idx == self.graph.next_edge_idx:
			raise StopIteration()

		while idx < self.graph.next_edge_idx and not self.graph.edge_info[idx].exists:
			idx += 1

		if idx >= self.graph.next_edge_idx:
			self.idx = idx
			raise StopIteration()

		self.idx = idx + 1
		if self.data is True:
			val = None
			if idx in self.graph.edge_data_lookup:
				val = self.graph.edge_data_lookup[idx]
			if not self.objs:
				return idx, val
			else:
				return [self.graph.node_obj_lookup[self.graph.edge_info[idx].nlist[i]] for i in range(self.graph.edge_info[idx].num_nodes)], val
		else:
			if not self.objs:
				return idx
			else:
				return [self.graph.node_obj_lookup[self.graph.edge_info[idx].nlist[i]] for i in range(self.graph.edge_info[idx].num_nodes)]

	def __iter__(AllEdgeIterator self):
		return self

cdef class NodeEdgeIterator:
	cdef bool data
	cdef HyperGraph graph
	cdef int nidx
	cdef int idx
	cdef bool endpoints

	def __cinit__(NodeEdgeIterator self,HyperGraph graph,nidx,data=False,endpoints=False):
		self.graph = graph
		self.nidx = nidx
		self.data = data
		self.idx = 0
		self.endpoints = endpoints

	def __next__(NodeEdgeIterator self):
		cdef int idx = self.idx
		cdef int* elist
		cdef int i

		if idx >= self.graph.node_info[self.nidx].degree:
			raise StopIteration

		num_edges = self.graph.node_info[self.nidx].degree
		elist = self.graph.node_info[self.nidx].elist

		self.idx = idx + 1

		if self.data is True:
			val = None
			if elist[idx] in self.graph.edge_data_lookup:
				val = self.graph.edge_data_lookup[elist[idx]]
			if not self.endpoints:
				return elist[idx], val
			else:
				idx = elist[idx]

				return [self.graph.node_obj_lookup[self.graph.edge_info[idx].nlist[i]] for i in range(self.graph.edge_info[idx].num_nodes)], val
		else:
			if not self.endpoints:
				return elist[idx]
			else:
				idx = elist[idx]

				return [self.graph.node_obj_lookup[self.graph.edge_info[idx].nlist[i]] for i in range(self.graph.edge_info[idx].num_nodes)]

	def __iter__(NodeEdgeIterator self):
		return self

cdef class SomeEdgeIterator:
	"""
	TODO: Boost performance by making touched_edges a C-level set
	"""
	cdef bool data
	cdef HyperGraph graph
	cdef touched_edges
	cdef nbunch_iter
	cdef edge_iter
	cdef bool endpoints

	def __cinit__(SomeEdgeIterator self,HyperGraph graph,nbunch,data=False,endpoints=False):
		self.graph = graph
		self.nbunch_iter = iter(nbunch)
		self.data = data
		self.edge_iter = None
		self.touched_edges = set()
		self.endpoints = endpoints

		# setup the first iterator
		if len(nbunch) > 0:
			curr_nidx = self.nbunch_iter.next()
			self.edge_iter = NodeEdgeIterator(self.graph,curr_nidx,self.which_degree,self.data)
		else:
			self.edge_iter = None

	def __next__(SomeEdgeIterator self):
		cdef int i
		while True:
			if self.edge_iter is None:
				raise StopIteration
			else:
				try:
					result = self.edge_iter.next()
					if self.data:
						if result[0] in self.touched_edges:
							continue
						self.touched_edges.add(result[0])

						if not self.endpoints:
							return result
						else:
							return [self.graph.node_obj_lookup[self.graph.edge_info[result[0]].nlist[i]] for i in range(self.graph.edge_info[result[0]].num_nodes)], result[1]
					else:
						if result in self.touched_edges:
							continue
						self.touched_edges.add(result)

						if not self.endpoints:
							return result
						else:
							return [self.graph.node_obj_lookup[self.graph.edge_info[result].nlist[i]] for i in range(self.graph.edge_info[result].num_nodes)]
				except StopIteration:
					self.edge_iter = None
					curr_nidx = self.nbunch_iter.next()
					self.edge_iter = NodeEdgeIterator(self.graph,curr_nidx,self.which_degree,self.data)

	def __iter__(SomeEdgeIterator self):
		return self

cdef class NeighborIterator:
	"""
	TODO: Speed boost by making touched_nodes a C-level set.
	"""
	cdef NodeEdgeIterator inner_iter
	cdef data
	cdef deg
	cdef int nidx
	cdef int eidx, nlist_idx
	cdef HyperGraph G
	cdef touched_nodes
	cdef use_nobjs

	def __cinit__(NeighborIterator self, HyperGraph G, int nidx,which_degree,data,use_nobjs):
		self.inner_iter = NodeEdgeIterator(G,nidx,which_degree,False)
		self.data = data
		self.deg = which_degree
		self.nidx = nidx
		self.G = G
		self.touched_nodes = set()
		self.use_nobjs = use_nobjs
		
		self.eidx = -1
		self.nlist_idx = 0

	def __next__(NeighborIterator self):
		cdef int eid
		cdef int nlidx
		
		while True:
			if self.eidx == -1 or self.nlist_idx >= self.G.edge_info[self.eidx].num_nodes:
				eid = self.inner_iter.next()
				self.eidx = eid
				nlidx = 0
				self.nlist_idx = 1
			else:
				eid = self.eidx
				nlidx = self.nlist_idx
				self.nlist_idx += 1
			
			if self.G.edge_info[eid].nlist[nlidx] == self.nidx:
				continue
					
			if self.data:
				val = None
				if self.G.edge_info[eid].nlist[nlidx] in self.touched_nodes:
					continue
				self.touched_nodes.add(self.G.edge_info[eid].nlist[nlidx])	
				
				if self.use_nobjs:
					return self.G.node_obj_lookup[self.G.edge_info[eid].nlist[nlidx]], self.G.node_data_(self.G.edge_info[eid].nlist[nlidx])
				else:
					return self.G.edge_info[eid].nlist[nlidx], self.G.node_data_(self.G.edge_info[eid].nlist[nlidx])
			else:	
				if self.G.edge_info[eid].nlist[nlidx] in self.touched_nodes:
					continue
				self.touched_nodes.add(self.G.edge_info[eid].nlist[nlidx])

				if self.use_nobjs:
					return self.G.node_obj_lookup[self.G.edge_info[eid].nlist[nlidx]]
				else:
					return self.G.edge_info[eid].nlist[nlidx]

	def __iter__(NeighborIterator self):
		return self

cdef class SomeNeighborIterator:
	"""
	TODO: Boost performance by making touched_nodes a C-level set
	"""
	cdef bool data
	cdef HyperGraph graph
	cdef int idx
	cdef touched_nodes
	cdef nbunch_iter
	cdef neighbor_iter
	cdef use_nobjs

	def __cinit__(SomeNeighborIterator self,HyperGraph graph,nbunch,data,use_nobjs):
		self.graph = graph
		self.nbunch_iter = iter(nbunch)
		self.data = data
		self.idx = 0
		self.neighbor_iter = None
		self.touched_nodes = set()
		self.use_nobjs = use_nobjs

		# setup the first iterator
		if len(nbunch) > 0:
			curr_nidx = self.nbunch_iter.next()
			self.neighbor_iter = NeighborIterator(self.graph,curr_nidx,self.data,self.use_nobjs)
		else:
			self.neighbor_iter = None

	def __next__(SomeNeighborIterator self):
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
					self.neighbor_iter = NeighborIterator(self.graph,curr_nidx,self.which_degree,self.data,self.use_nobjs)

	def __iter__(SomeNeighborIterator self):
		return self		
