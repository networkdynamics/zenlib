cimport cfiboheap

cdef inline object convert_fibheap_el_to_pycapsule(cfiboheap.fibheap_el* element)
cdef inline cfiboheap.fibheap_el* convert_pycapsule_to_fibheap_el(object element)
	
cdef class FiboHeap:

	# attributes
	cdef cfiboheap.fibheap* treeptr
	
	cpdef object insert(FiboHeap self, double key, object data=*)
	cdef  void* _insert(FiboHeap self, double key, void* data=*)
	
	cpdef object peek(FiboHeap self)
	cdef  void* _peek(FiboHeap self)
	
	cpdef object extract(FiboHeap self)
	cdef  void* _extract(FiboHeap self)
	
	cpdef double get_min_key(FiboHeap self)
	cdef  double _get_min_key(FiboHeap self)
	
	cpdef object decrease_key(FiboHeap self, object element, double newKey)
	cdef  void* _decrease_key(FiboHeap self, void* element, double newKey)
	
	cpdef object replace_data(FiboHeap self, object element, object data)
	cdef  void* _replace_data(FiboHeap self, cfiboheap.fibheap_el* element, void* data)
	
	cpdef object delete(FiboHeap self, object element)
	cdef  void* _delete(FiboHeap self, cfiboheap.fibheap_el* element)
	
	cpdef heap_union(FiboHeap self, object heap)
	cdef _heap_union(FiboHeap self, cfiboheap.fibheap* heap)
	
	cpdef int get_node_count(FiboHeap self)
	cdef int _get_node_count(FiboHeap self)
	
	cpdef object get_node_data(FiboHeap self, object element)
	cdef void* _get_node_data(FiboHeap self, cfiboheap.fibheap_el* element)
	
	cpdef double get_node_key(FiboHeap self, object element)
	cdef double _get_node_key(FiboHeap self, cfiboheap.fibheap_el* element)