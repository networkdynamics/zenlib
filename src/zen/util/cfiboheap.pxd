cdef extern from "fib.h":	
	cdef struct fibheap:
		pass
		
	cdef struct fibheap_el:
		pass
				
	fibheap *			fh_makekeyheap()
	fibheap_el *		fh_insertkey(fibheap *, double, void *)
	double 				fh_minkey(fibheap *)
	void * 				fh_replacekey(fibheap *, fibheap_el *, double)
	int					get_node_count(fibheap *)
	void* 				get_fibheap_el_data(fibheap_el *)
	double	 			get_fibheap_el_key(fibheap_el *)
	
	
	void *				fh_extractmin(fibheap *)
	void *				fh_min(fibheap *)
	void *				fh_replacedata(fibheap *, fibheap_el *, void *)
	void *				fh_delete(fibheap *, fibheap_el *)
	void 				fh_deleteheap(fibheap *)
	fibheap *			fh_union(fibheap *, fibheap *)

		