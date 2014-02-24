"""
The ``zen.BipartiteGraph`` class provides an undirected `bipartite graph <http://en.wikipedia.org/wiki/Bipartite_graph>`_.  
Each node must belong to one of two classes: ``U`` or ``V``.  Edges must connect nodes in different classes.  
Besides this, the graph	object follows the same rules and exposes the same functions as the standard undirected
graph.

Because this class extends :py:class:`zen.Graph`, it implements the same methods with a handful of 
exceptions:

	* Nodes must be explicitly added as either a ``U`` node or a ``V`` node.  To enforce this, 
	  :py:meth:`.add_node` raises a :py:exc:`zen.ZenException`.  In place of this method,
	  the following options exist
		
	  * :py:meth:`.add_node_by_class`
	  * :py:meth:`.add_u_node`
	  * :py:meth:`.add_v_node`
		
	* Edges must have one endpoint in the ``U`` set and one in the ``V`` set.  Thus if 
	  :py:meth:`.add_edge` is called with nodes that are not members of the graph,
	  it is not explicit which node belongs to which set.  A :py:exc:`zen.ZenException` is
	  now raised by this method if the endpoints are not part of the graph.
"""

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
	
	def __init__(self,**kwargs):
		"""
		Create a new :py:class:`BipartiteGraph` object.
		
		**KwArgs**:
		
		  * ``node_capacity [=100]`` (int): the initial number of nodes this graph has space to hold.
		  * ``edge_capacity [=100]`` (int): the initial number of edges this graph has space to hold.
		  * ``edge_list_capacity [=5]`` (int): the initial number of edges that each node is allocated space for initially.
		"""
		
		Graph.__init__(self,**kwargs)
		
		self.node_assignments = <int*> stdlib.malloc(sizeof(int)*self.node_capacity)
		self.node_assignments_capacity = self.node_capacity
		self.u_nodes = set()
		self.v_nodes = set()
	
	cpdef copy(BipartiteGraph self):
		cdef BipartiteGraph G = BipartiteGraph()
		
		# copy the raw graph content in
		self.__copy_graph_self_into(G)
		
		# copy the bipartite-specific stuff
		G.node_assignments_capacity = self.node_assignments_capacity
		stdlib.free(G.node_assignments)
		G.node_assignments = <int*> stdlib.malloc(sizeof(int) * G.node_assignments_capacity)
		for i in range(G.node_assignments_capacity):
			G.node_assignments[i] = self.node_assignments_capacity
			
		G.u_nodes = set(self.u_nodes)
		G.v_nodes = set(self.v_nodes)
			
		# done
		return G
	
	cpdef int add_node_by_class(self,bint as_u,nobj=None,data=None) except -1:
		"""
		Add a node to either the ``U`` or the ``V`` class.  
		
		**Args**:
		
			* ``as_u`` (boolean): if ``True``, then the node is added to the ``U`` class.  Otherwise, the node
			  is added to the ``V`` class.
			* ``nobj [=None]``: the object to be associated as an identifier for the node created.  If ``None``,
			  then no object will be associated as an idenifier for the node.
			* ``data [=None]``: the data object to associate with the node created.
		
		**Returns**:
			``integer``. The index for the node created.
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
	
	cpdef int add_u_node(self,nobj=None,data=None) except -1:
		"""
		Add a node to the ``U`` class.  
		
		**Args**:
		
			* ``nobj [=None]``: the object to be associated as an identifier for the node created.  If ``None``,
			  then no object will be associated as an idenifier for the node.
			* ``data [=None]``: the data object to associate with the node created.
		
		**Returns**:
			``integer``. The index for the node created.
		"""
		return self.add_node_by_class(True,nobj,data)
		
	cpdef int add_v_node(self,nobj=None,data=None) except -1:
		"""
		Add a node to the ``V`` class.  
		
		**Args**:
		
			* ``nobj [=None]``: the object to be associated as an identifier for the node created.  If ``None``,
			  then no object will be associated as an idenifier for the node.
			* ``data [=None]``: the data object to associate with the node created.
		
		**Returns**:
			``integer``. The index for the node created.
		"""
		return self.add_node_by_class(False,nobj,data)
	
	cpdef int add_node(self,nobj=None,data=None) except -1:
		"""
		This method for adding arbitrary nodes to a :py:class:`zen.Graph` is not supported in 
		the :py:class:`zen.BipartiteGraph` graph because nodes must	be added to one of two classes.
		
		If called, this method raises a :py:exc:`zen.ZenException`.
		"""
		raise ZenException, 'Bipartite.add_node method is not supported. Use add_node_by_class, add_u_node, or add_v_node instead.'
			
	cpdef rm_node(self,nobj):
		"""
		Remove the node associated with node object ``nobj``.
		"""
		self.rm_node_(self.node_idx_lookup[nobj])

	cpdef rm_node_(self,int nidx):
		"""
		Remove the node with index ``nidx``.
		"""
		Graph.rm_node_(self,nidx)
		
		if nidx in self.u_nodes:
			self.u_nodes.remove(nidx)
		else:
			self.v_nodes.remove(nidx)
	
	cpdef int add_edge(self, u, v, data=None, double weight=1) except -1:
		"""
		Add an edge between nodes ``u`` and ``v`` (node objects).
		
		.. note::
			Unlike it's parent method, :py:meth:`zen.Graph.add_edge`, this method does not automatically create
			nodes for objects that are not currently in the graph.  This change in behavior is due to the fact that
			all nodes must belong to either the ``U`` or ``V`` classes.  In the absence of explicit indication of which
			node should be added to which class, this convenience cannot be provided.
			
		**Args**:

			* ``u``: one endpoint of the graph. 
			* ``v``: another endpoint of the graph.
			* ``data [=None]``: an optional data object to associate with the edge
			* ``weight [=1]`` (float): the weight of the edge.

		**Returns**:
			``integer``. The index for the newly created edge.

		**Raises**:
			:py:exc:`ZenException`: if the edge already exists in the graph or if ``u`` and ``v`` belong to the
			same node class.
		"""
		if u not in self.node_idx_lookup:
			raise ZenException, '%s is not a valid node object in the graph' % str(u)
		elif v not in self.node_idx_lookup:
			raise ZenException, '%s is not a valid node object in the graph' % str(v)

		return Graph.add_edge(self,u,v,data,weight)
	
	cpdef int add_edge_(self, int u, int v, data=None, double weight=1) except -1:
		"""
		Add an edge to the graph.			
		
		**Args**:
		
			* ``u`` (int): one endpoint of the graph. This is a node index.
			* ``v`` (int): another endpoint of the graph. This is a node index.
			* ``data [=None]``: an optional data object to associate with the edge
			* ``weight [=1]`` (float): the weight of the edge.
			
		**Returns**:
			``integer``. The index for the newly created edge.
			
		**Raises**:
			:py:exc:`ZenException`: if the edge already exists in the graph, if either of the node indices are invalid,
			or if both nodes belong to the same node class.
		"""
		if self.is_in_U_(u) == self.is_in_U_(v):
			raise ZenException, 'Two nodes from the same class cannot be connected by an edge in a bipartite graph'
			
		return Graph.add_edge_(self,u,v,data,weight)
		
	cpdef bint is_in_U(self, nobj) except -1:
		"""
		Return ``True`` if the node identified by object ``nobj`` is in the ``U`` node class.
		"""
		return self.is_in_U_(self.node_idx_lookup[nobj])
		
	cpdef bint is_in_U_(self, int nidx) except -1:
		"""
		Return ``True`` if the node with index ``nidx`` is in the ``U`` node class.
		"""
		return self.node_assignments[nidx]
	
	cpdef bint is_in_V(self, nobj) except -1:
		"""
		Return ``True`` if the node identified by object ``nobj`` is in the ``V`` node class.
		"""
		return self.is_in_V_(self.node_idx_lookup[nobj])

	cpdef bint is_in_V_(self, int nidx) except -1:
		"""
		Return ``True`` if the node with index ``nidx`` is in the ``V`` node class.
		"""
		return not self.node_assignments[nidx]
	
	cpdef U(self):
		"""
		Return a :py:class:`list` containing the node objects of the nodes in the ``U`` set.
		"""
		return list([self.node_obj_lookup[u] for u in self.u_nodes])
	
	cpdef V(self):
		"""
		Return a :py:class:`list` containing the node objects of the nodes in the ``V`` set.
		"""		
		return list([self.node_obj_lookup[u] for u in self.v_nodes])
		
	cpdef U_(self):
		"""
		Return a numpy array containing the node indices of the nodes in the ``U`` set.
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
		Return a numpy array containing the node indices of the nodes in the ``V`` set.
		"""
		cdef np.ndarray[np.int_t, ndim=1] v_nodes = np.zeros(len(self.v_nodes),np.int)
		cdef int i = 0
		cdef int n
		for n in self.v_nodes:
			v_nodes[i] = n
			i += 1
			
		return v_nodes
		
	cpdef uv_endpoints_(self,int eidx):
		"""
		Return the endpoints of the edge with index ``eidx`` in u-v order.  The node indicies are returned
		by this function call.
		
		This method contrasts with the :py:meth:`BipartiteGraph.endpoints_` method which returns the
		endpoints in an, effectively, arbitrary order.
		"""
		xi = self.edge_info[eidx].u
		yi = self.edge_info[eidx].v
		
		if self.node_assignments[xi]:
			return xi,yi
		else:
			return yi,xi
			
	cpdef uv_endpoints(self,int eidx):
		"""
		Return the endpoints of the edge with index ``eidx`` in u-v order.  The node objects are returned
		by this function call.
		
		This method contrasts with the :py:meth:`BipartiteGraph.endpoints` method which returns the
		endpoints in an, effectively, arbitrary order.
		"""
		ui,vi = self.uv_endpoints_(eidx)
		
		return self.node_object(ui), self.node_object(vi)

	cpdef uv_edges(self,nobj=None,bint data=False,bint weight=False):
		"""
		Return a list of edges in the graph.
	
		Unlike the list returned by :py:meth:`Graph.edges`, this list will contain all edges in the graph, each
		edge as the tuple ``(u,v)``, where ``u`` is the u-class node object and ``v`` is the v-class node object.
		
		The arguments are exactly the same as for :py:meth:`Graph.edges`.
	
		**Args**:
		
			* ``nobj [=None]``: if ``nobj`` is specified (not ``None``), then only the edges touching the node with 
				object ``nobj`` are included in the list.
	
			* ``data [=False]`` (boolean): if ``True``, then the data object associated with the edge
			 	is added into the tuple returned for each edge (e.g., ``(u,v,d)``).
	
			* ``weight [=False]`` (boolean): 	if ``True``, then the weight of the edge is added
				into the tuple returned for each edge (e.g., ``(u,v,w)`` or ``(u,v,data,w)`` depending on the 
				value of the ``data`` argument).
		"""
		cdef int num_edges
		cdef int* elist
		cdef int i
		cdef nidx = -1
		cdef nidx2 = -1
		cdef int u,v
		
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
					
					u = self.edge_info[i].u
					v = self.edge_info[i].v
					
					if self.node_assignments[v]:
						v = self.edge_info[i].u
						u = self.edge_info[i].v
						
						
					if data is True:
						if weight is True:
							result.append( (self.node_obj_lookup[u],self.node_obj_lookup[v],self.edge_data_(i),self.edge_info[i].weight) )
						else:
							result.append( (self.node_obj_lookup[u],self.node_obj_lookup[v],self.edge_data_(i)) )
					else:
						if weight:
							result.append( (self.node_obj_lookup[u],self.node_obj_lookup[v], self.edge_info[i].weight) )
						else:
							result.append( (self.node_obj_lookup[u],self.node_obj_lookup[v]) )
					idx += 1
					
			return result
		else:
			idx = 0
			num_edges = self.node_info[nidx].degree
			elist = self.node_info[nidx].elist
			for i in range(num_edges):
				if self.edge_info[elist[i]].u not in self.node_obj_lookup or self.edge_info[elist[i]].v not in self.node_obj_lookup:
					raise ZenException, 'Edge (idx=%d) does not have endpoints with node objects' % i
					
				u = self.edge_info[elist[i]].u
				v = self.edge_info[elist[i]].v
				
				if self.node_assignments[v]:
					v = self.edge_info[elist[i]].u
					u = self.edge_info[elist[i]].v
						
				if data is True:
					if weight is True:
						result.append( (self.node_obj_lookup[u],self.node_obj_lookup[v],self.edge_data_(elist[i]), self.edge_info[elist[i]].weight) )
					else:
						result.append( (self.node_obj_lookup[u],self.node_obj_lookup[v],self.edge_data_(elist[i])) )
				else:
					if weight:
						result.append( (self.node_obj_lookup[u],self.node_obj_lookup[v],self.edge_info[elist[i]].weight) )
					else:
						result.append( (self.node_obj_lookup[u],self.node_obj_lookup[v]) )
				idx += 1
			
			return result

	cpdef uv_edges_iter(self,nobj=None,bint data=False,bint weight=False):
		"""
		Return an iterator over edges in the graph.
		
		Unlike the iterator returned by :py:meth:`Graph.edges_iter`, this one will yield all edges in the graph, each
		edge as the tuple ``(u,v)``, where ``u`` is the u-class node object and ``v`` is the v-class node object.
		
		The arguments are exactly the same as for :py:meth:`Graph.edges_iter`.
		
		**Args**:
			
			* ``nobj [=None]``: if ``nobj`` is specified (not ``None``), then the edges touching the node with 
				object ``nobj`` are iterated over.
		
			* ``data [=False]`` (boolean): if ``True``, then the iterator adds object associated with the edge
			 	into the tuple returned (e.g., ``(u,v,d)``).
		
			* ``weight [=False]`` (boolean): 	if ``True``, then the iterator adds the weight of the edge
				into the tuple returned (e.g., ``(u,v,w)`` or ``(u,v,data,w)`` depending on the value of the ``data`` argument).
		"""
		iterator = self.edges_iter(nobj,data,weight)
		
		uv_iterator = UVEdgeIterator(self,iterator)
		return uv_iterator
		
class UVEdgeIterator:
	"""
	This iterator wraps another edge iterator.  It ensures that the order in which node objects are returned is u-v.
	"""
	def __init__(self,G,iterator):
		self.G = G
		self.iterator = iterator
		
	def next(self):
		result = next(self.iterator)
		
		x,y = result[0:2]
		if self.G.is_in_U(y):
			result = tuple(y,x,*result[2:])
			
		return result
		
	def __iter__(self):
		return self
		
	