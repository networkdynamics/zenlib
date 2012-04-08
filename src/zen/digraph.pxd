from cpython cimport bool

import numpy as np
cimport numpy as np



cdef struct NodeInfo:
	bint exists
	
	int indegree
	int* inelist
	int in_capacity
	
	int outdegree
	int* outelist
	int out_capacity
	
cdef struct EdgeInfo:
	bint exists
	int src
	int tgt
	double weight
	
cdef class DiGraph:
	
	# attributes
	cdef public double node_grow_factor
	cdef public double edge_list_grow_factor
	cdef public double edge_grow_factor
	
	cdef long num_changes
	cdef readonly int num_nodes
	cdef readonly int node_capacity
	cdef int next_node_idx
	cdef readonly int max_node_idx
	cdef NodeInfo* node_info
	cdef node_idx_lookup
	cdef node_obj_lookup
	cdef node_data_lookup
	cdef int first_free_node
	
	cdef readonly int num_edges
	cdef readonly int edge_capacity
	cdef int next_edge_idx
	cdef readonly int max_edge_idx
	cdef EdgeInfo* edge_info
	cdef edge_data_lookup
	cdef int first_free_edge
	
	cdef readonly int edge_list_capacity
	
	# methods
	cdef inner_validate(self,bint verbose)
	
	cpdef np.ndarray[np.double_t] matrix(self)
	
	cpdef copy(DiGraph self)
	
	cpdef is_directed(DiGraph self)
	
	cpdef bool is_compact(DiGraph self)
	
	cpdef compact(DiGraph self)
	
	cpdef skeleton(self,data_merge_fxn=*,weight_merge_fxn=*)
	
	cpdef reverse(self)
	
	cpdef np.ndarray[np.int_t] add_nodes(self,int num_nodes,node_obj_fxn=*)
	
	cpdef int add_node(DiGraph self,nobj=*,data=*)
	
	cdef add_to_free_node_list(self,int nidx)
	
	cdef remove_from_free_node_list(self,int nidx)
	
	cpdef add_node_x(DiGraph self,int node_idx,int in_edge_list_capacity,int out_edge_list_capacity,nobj,data)
	
	cpdef int node_idx(DiGraph self,nobj) except -1
	
	cpdef node_object(DiGraph self,int nidx)
	
	cpdef set_node_object(self,curr_node_obj,new_node_obj)
	
	cpdef set_node_object_(self,node_idx,new_node_obj)
	
	cpdef set_node_data(DiGraph self,nobj,data)
	
	cpdef node_data(DiGraph self,nobj)
	
	cpdef set_node_data_(DiGraph self,int nidx,data)
	
	cpdef node_data_(DiGraph self,int nidx)
	
	cpdef nodes_iter(DiGraph self,data=*)
	
	cpdef nodes_iter_(DiGraph self,obj=*,data=*)
	
	cpdef nodes(DiGraph self,data=*)
		
	cpdef nodes_(DiGraph self,obj=*,data=*)
	
	cpdef rm_node(DiGraph self,nobj)
	
	cpdef rm_node_(DiGraph self,int nidx)
	
	cpdef degree(DiGraph self,nobj)

	cpdef degree_(DiGraph self,int nidx)
	
	cpdef in_degree(DiGraph self,nobj)
	
	cpdef in_degree_(DiGraph self,int nidx)
	
	cpdef out_degree(DiGraph self,nobj)
	
	cpdef out_degree_(DiGraph self,int nidx)
	
	cpdef int add_edge(DiGraph self, src, tgt, data=*, double weight=*) except -1
	
	cpdef int add_edge_(DiGraph self, int src, int tgt, data=*, double weight=*) except -1
	
	cdef add_to_free_edge_list(self,int eidx)
	
	cdef remove_from_free_edge_list(self,int eidx)
	
	cpdef int add_edge_x(DiGraph self, int eidx, int src, int tgt, data, double weight) except -1
	
	cdef __insert_edge_into_outelist(DiGraph self, int src, int eidx, int tgt)
	
	cdef __insert_edge_into_inelist(DiGraph self, int tgt, int eidx, int src)
	
	cdef int find_inelist_insert_pos(DiGraph self, int* elist, int elist_len, int nidx)
	
	cdef int find_outelist_insert_pos(DiGraph self, int* elist, int elist_len, int nidx)
	
	cpdef rm_edge(DiGraph self,src,tgt)
	
	cpdef rm_edge_(DiGraph self,int eidx)
	
	cdef __remove_edge_from_outelist(DiGraph self, int src, int eidx, int tgt)
	
	cdef __remove_edge_from_inelist(DiGraph self, int tgt, int eidx, int src)
	
	cpdef endpoints(DiGraph self,int eidx)
	
	cpdef endpoints_(DiGraph self,int eidx)
	
	cpdef endpoint(DiGraph self,int eidx,u)
	
	cpdef int endpoint_(DiGraph self,int eidx,int u) except -1
		
	cpdef src(DiGraph self,int eidx)
	
	cpdef int src_(DiGraph self,int eidx) except -1
	
	cpdef tgt(DiGraph self,int eidx)
	
	cpdef int tgt_(DiGraph self,int eidx) except -1	
		
	cpdef set_weight(DiGraph self,u,v,double w)
	
	cpdef set_weight_(DiGraph self,int eidx,double w)
	
	cpdef double weight(DiGraph self,u,v)
	
	cpdef double weight_(DiGraph self,int eidx)
		
	cpdef set_edge_data(DiGraph self,src,tgt,data)
		
	cpdef edge_data(DiGraph self,src,tgt)
	
	cpdef set_edge_data_(DiGraph self,int eidx,data)
	
	cpdef edge_data_(DiGraph self,int eidx,int dest=*)
	
	cpdef bool has_edge(DiGraph self,src,tgt)
		
	cpdef bool has_edge_(DiGraph self,int src,int tgt)
	
	cpdef edge_idx(DiGraph self, src, tgt, data=*)
	
	cpdef edge_idx_(DiGraph self, int src, int tgt, data=*)
	
	cpdef edges_iter(DiGraph self,nobj=*,data=*,weight=*)
	
	cpdef edges_iter_(DiGraph self,int nidx=*,data=*,weight=*)
	
	cpdef edges(DiGraph self,nobj=*,data=*,weight=*)
		
	cpdef edges_(DiGraph self,int nidx=*,data=*,weight=*)

	cpdef in_edges_iter(DiGraph self,nobj,data=*,weight=*)

	cpdef in_edges_iter_(DiGraph self,int nidx,data=*,weight=*)
		
	cpdef in_edges(DiGraph self,nobj,data=*,weight=*)

	cpdef in_edges_(DiGraph self,int nidx,data=*,weight=*)
		
	cpdef out_edges_iter(DiGraph self,nobj,data=*,weight=*)
	
	cpdef out_edges_iter_(DiGraph self,int nidx,data=*,weight=*)

	cpdef out_edges(DiGraph self,nobj,data=*,weight=*)

	cpdef out_edges_(DiGraph self,int nidx,data=*,weight=*)

	cpdef grp_edges_iter(DiGraph self,nbunch,data=*,weight=*)

	cpdef grp_in_edges_iter(DiGraph self,nbunch,data=*,weight=*)

	cpdef grp_out_edges_iter(DiGraph self,nbunch,data=*,weight=*)
	
	cpdef grp_edges_iter_(DiGraph self,nbunch,data=*,weight=*)

	cpdef grp_in_edges_iter_(DiGraph self,nbunch,data=*,weight=*)

	cpdef grp_out_edges_iter_(DiGraph self,nbunch,data=*,weight=*)
	
	cpdef neighbors(DiGraph self,nobj,data=*)
	
	cpdef neighbors_(DiGraph self,int nidx,obj=*,data=*)
	
	cpdef neighbors_iter(DiGraph self,nobj,data=*)
	
	cpdef neighbors_iter_(DiGraph self,int nidx,obj=*,data=*)
		
	cpdef in_neighbors_iter(DiGraph self,nobj,data=*)
	
	cpdef in_neighbors_iter_(DiGraph self,int nidx,obj=*,data=*)
	
	cpdef in_neighbors(DiGraph self,nobj,data=*)
	
	cpdef in_neighbors_(DiGraph self,int nidx,obj=*,data=*)
		
	cpdef out_neighbors_iter(DiGraph self,nobj,data=*)
		
	cpdef out_neighbors_iter_(DiGraph self,int nidx,obj=*,data=*)
	
	cpdef out_neighbors(DiGraph self,nobj,data=*)
	
	cpdef out_neighbors_(DiGraph self,int nidx,obj=*,data=*)
	
	cpdef grp_neighbors_iter(DiGraph self,nbunch,data=*)

	cpdef grp_in_neighbors_iter(DiGraph self,nbunch,data=*)

	cpdef grp_out_neighbors_iter(DiGraph self,nbunch,data=*)
	
	cpdef grp_neighbors_iter_(DiGraph self,nbunch,obj=*,data=*)

	cpdef grp_in_neighbors_iter_(DiGraph self,nbunch,obj=*,data=*)

	cpdef grp_out_neighbors_iter_(DiGraph self,nbunch,obj=*,data=*)