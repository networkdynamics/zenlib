from graph cimport Graph

cdef class BipartiteGraph(Graph):
	
	cdef int* node_assignments
	cdef int node_assignments_capacity
	cdef u_nodes
	cdef v_nodes

	# node addition
	cpdef int add_node_by_class(self,bint is_u,nobj=*,data=*) except -1
	cpdef int add_u_node(self,nobj=*,data=*) except -1
	cpdef int add_v_node(self,nobj=*,data=*) except -1
	cpdef int add_node(self,nobj=*,data=*) except -1
	
	# node removal
	cpdef rm_node(self,nobj)
	cpdef rm_node_(self,int nidx)
	
	# edge addition
	cpdef int add_edge(self, u, v, data=*, double weight=*) except -1
	cpdef int add_edge_(self, int u, int v, data=*, double weight=*) except -1
	
	# accessing U & V classes
	cpdef bint is_in_U(self,nobj) except -1
	cpdef bint is_in_U_(self,int nidx) except -1
	cpdef bint is_in_V(self,nobj) except -1
	cpdef bint is_in_V_(self,int nidx) except -1
	cpdef U(self)
	cpdef V(self)
	cpdef U_(self)
	cpdef V_(self)
	
	# edge accessors
	cpdef uv_endpoints_(self,int eidx)
	cpdef uv_endpoints(self,int eidx)