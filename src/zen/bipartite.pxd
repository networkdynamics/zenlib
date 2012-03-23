from graph cimport Graph

cdef class BipartiteGraph(Graph):
	
	cdef int* node_assignments
	cdef int node_assignments_capacity
	cdef u_nodes
	cdef v_nodes

	# node addition
	cpdef int add_node_by_class(self,int is_u,nobj=*,data=*)
	cpdef int add_u_node(self,nobj=*,data=*)
	cpdef int add_v_node(self,nobj=*,data=*)
	cpdef int add_node(self,nobj=*,data=*)
	
	# node removal
	cpdef rm_node(self,nobj)
	cpdef rm_node_(self,int nidx)
	
	# edge addition
	cpdef int add_edge_(self, int u, int v, data=*, double weight=*) except -1
	
	# accessing U & V classes
	cpdef int is_in_U(self,nobj)
	cpdef int is_in_U_(self,int nidx)
	cpdef U(self)
	cpdef V(self)
	cpdef U_(self)
	cpdef V_(self)