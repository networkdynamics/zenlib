from cpython cimport bool
from graph cimport Graph

cdef struct NodeInfo:
	bint exists
	
	int degree
	int* elist
	int capacity
	
cdef struct EdgeInfo:
	bint exists
	int* nlist
	int num_nodes
	int capacity
	
cdef class HyperGraph:
	
	cdef float node_grow_factor
	cdef float edge_list_grow_factor
	cdef float edge_grow_factor

	cdef int num_nodes
	cdef int num_edges
	
	cdef int node_capacity
	cdef int edge_list_capacity
	cdef NodeInfo* node_info
	cdef int next_node_idx
	cdef node_obj_lookup
	cdef node_data_lookup
	cdef node_idx_lookup
	
	
	cdef int edge_capacity
	cdef EdgeInfo* edge_info
	cdef int next_edge_idx
	cdef edge_idx_lookup
	cdef edge_data_lookup

	cpdef Graph to_graph(HyperGraph self)
	
	cpdef bool is_directed(HyperGraph self)
	
	cpdef int add_node(HyperGraph self,nobj=*,data=*)

	cpdef is_valid_node_idx(HyperGraph self, int nidx)

	cpdef int node_idx(HyperGraph self,nobj)

	cpdef node_object(HyperGraph self,int nidx)

	cpdef node_data(HyperGraph self,nobj)

	cpdef node_data_(HyperGraph self,int nidx)

	cpdef nodes_iter(HyperGraph self,data=*)

	cpdef nodes_iter_(HyperGraph self,data=*)

	cpdef nodes(HyperGraph self,data=*)

	cpdef nodes_(HyperGraph self,data=*)

	cpdef rm_node(HyperGraph self,nobj)

	cpdef rm_node_(HyperGraph self,int nidx)

	cpdef degree(HyperGraph self,nobj)

	cpdef degree_(HyperGraph self,int nidx)

	cpdef add_edge(HyperGraph self, node_list, data=*)

	cpdef int add_edge_(HyperGraph self, node_list, data=*)

	cpdef rm_edge(HyperGraph self,src,tgt)

	cpdef rm_edge_(HyperGraph self,int eidx)

	cpdef card(HyperGraph self,nlist)
		
	cpdef card_(HyperGraph self,nlist)

	cpdef endpoints(HyperGraph self,int eidx)

	cpdef endpoints_(HyperGraph self,int eidx)

	cpdef edge_data(HyperGraph self,nlist)

	cpdef edge_data_(HyperGraph self,nlist)

	cpdef has_edge(HyperGraph self,nlist)

	cpdef bool has_edge_(HyperGraph self,nlist)

	cpdef edge_idx(HyperGraph self, nlist, data=*)

	cpdef edge_idx_(HyperGraph self, nlist, data=*)

	cpdef edges_iter(HyperGraph self,nobj=*,data=*)

	cpdef edges_iter_(HyperGraph self,int nidx=*,data=*)

	cpdef edges(HyperGraph self,nobj=*,data=*)

	cpdef edges_(HyperGraph self,int nidx=*,data=*)

	cpdef grp_edges_iter(HyperGraph self,nbunch,data=*)

	cpdef grp_edges_iter_(HyperGraph self,nbunch,data=*)
		
	cpdef is_neighbor(HyperGraph self,nobj1,nobj2)

	cpdef is_neighbor_(HyperGraph self, int nidx1, int nidx2)

	cpdef neighbors(HyperGraph self,nobj,data=*)

	cpdef neighbors_(HyperGraph self,int nidx,data=*)

	cpdef neighbors_iter(HyperGraph self,nobj,data=*)

	cpdef neighbors_iter_(HyperGraph self,int nidx,data=*)

	cpdef grp_neighbors_iter(HyperGraph self,nbunch,data=*)

	cpdef grp_neighbors_iter_(HyperGraph self,nbunch,data=*)
