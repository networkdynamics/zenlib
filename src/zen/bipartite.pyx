from graph cimport Graph
from exceptions import *

import numpy as np
cimport numpy as np

cimport libc.stdlib as stdlib

__all__ = ['BipartiteGraph']



# TODO(druths): This implementation needs work
# * u_nodes and v_nodes need to be implemented in a pure c data structure... a linked list?
#
# * There should be iterator functions

cdef class BipartiteGraph(Graph):
	"""
	This class provides an undirected bipartite graph.  Each node must belong to one of two 
	classes: U or V.  Edges must connect nodes in different classes.  Besides this, the graph
	object follows the same rules and exposes the same functions as the standard undirected
	graph.
	"""
	def __cinit__(self):
		self.node_assignments = <int*> stdlib.malloc(sizeof(int)*self.node_capacity)
		self.node_assignments_capacity = self.node_capacity
		self.u_nodes = set()
		self.v_nodes = set()
	
	cpdef int add_node_by_class(self,int as_u,nobj=None,data=None):
		"""
		Add a node to either the U or the V class.  If as_u is True, then the
		node is added to the U class.  Otherwise, the node is added to the V class.
		"""
		nidx = Graph.add_node(self,nobj,data)
		
		# record the class of this node
		if self.node_capacity > self.node_assignments_capacity:
			self.node_assignments = <int*> stdlib.realloc(self.node_assignments,sizeof(int)*self.node_capacity)
			self.node_assignments_capacity = self.node_capacity
			
		self.node_assignments[nidx] = as_u
		if as_u:
			self.u_nodes.add(nidx)
		else:
			self.v_nodes.add(nidx)
			
		return nidx
	
	cpdef int add_u_node(self,nobj=None,data=None):
		return self.add_node_by_class(True,nobj,data)
		
	cpdef int add_v_node(self,nobj=None,data=None):
		return self.add_node_by_class(False,nobj,data)
	
	cpdef int add_node(self,nobj=None,data=None):
		"""
		This general method for adding nodes is not supported in Bipartite graph because nodes must
		be added to one of two classes.
		"""
		raise ZenException, 'add_node method is not supported in BipartiteGraph. Use add_node_by_class, add_u_node or add_v_node instead.'
			
	cpdef rm_node(self,nobj):
		"""
		Remove the node associated with node object nobj.
		"""
		self.rm_node_(self.node_idx_lookup[nobj])

	cpdef rm_node_(self,int nidx):
		Graph.rm_node_(self,nidx)
		
		if nidx in self.u_nodes:
			self.u_nodes.remove(nidx)
		else:
			self.v_nodes.remove(nidx)
	
	cpdef int add_edge_(self, int u, int v, data=None, double weight=1) except -1:
		
		if self.is_in_U_(u) == self.is_in_U_(v):
			raise ZenException, 'Two nodes from the same class cannot be connected by an edge in a bipartite graph'
			
		return Graph.add_edge_(self,u,v,data,weight)
		
	cpdef int is_in_U(self, nobj):
		return self.is_in_U_(self.node_idx_lookup[nobj])
		
	cpdef int is_in_U_(self, int nidx):
		return self.node_assignments[nidx]
	
	cpdef U(self):
		"""
		Return a list containing the node objects of the nodes in the U set.
		"""
		return list([self.node_obj_lookup[u] for u in self.u_nodes])
	
	cpdef V(self):
		"""
		Return a list containing the node objects of the nodes in the V set.
		"""		
		return list([self.node_obj_lookup[u] for u in self.v_nodes])
		
	cpdef U_(self):
		"""
		Return a numpy array containing the node indices of the nodes 
		in the U set.
		"""
		cdef np.ndarray[np.int_t, ndim=1] u_nodes = np.zeros(len(self.u_nodes),np.int)
		cdef int i = 0
		cdef int n
		for n in self.u_nodes:
			u_nodes[i] = n
			i += 1
			
		return u_nodes
		
	cpdef V_(self):
		"""
		Return a numpy array containing the node indices of the nodes 
		in the V set.
		"""
		cdef np.ndarray[np.int_t, ndim=1] v_nodes = np.zeros(len(self.v_nodes),np.int)
		cdef int i = 0
		cdef int n
		for n in self.v_nodes:
			v_nodes[i] = n
			i += 1
			
		return v_nodes
		