import numpy as np
cimport numpy as np

cdef class FibNode:
	cdef double key
	cdef unsigned int degree
	cdef bint mark
	cdef object data
	cdef FibNode parent
	cdef FibNode child
	cdef FibNode left
	cdef FibNode right
	cpdef get_key(FibNode self)
	cpdef print_node(FibNode self, indent=*, charr=*)
	cpdef object get_data(FibNode self)
	cpdef set_data(FibNode self, object data)
	cpdef FibNode get_left(FibNode self)
	cpdef FibNode get_right(FibNode self)
	cpdef FibNode get_parent(FibNode self)
	cpdef FibNode get_child(FibNode self)

	cdef __append_nodes(FibNode self, FibNode nodeX)
	cdef __link(FibNode self, FibNode nodeX)
	cdef __print_node_info(FibNode self)

cdef class FibHeap:
	cdef FibNode min
	cdef int nodeCount
	cpdef FibNode get_min(FibHeap self)
	cpdef insert_node(FibHeap self, FibNode node)
	cpdef FibNode extract_node(FibHeap self)
	cpdef object extract_min_node_data(FibHeap self)
	cpdef decrease_key(FibHeap self, FibNode node, double key)
	cpdef show_heap(FibHeap self)
	cpdef get_all_nodes(FibHeap self)

	cdef __consolidate(FibHeap self)
	cdef __cut(FibHeap self, FibNode nodeX, FibNode nodeY)
	cdef __cascading_cut(FibHeap self, FibNode node)
	cdef FibHeap __union(FibHeap self, FibHeap heap2)




