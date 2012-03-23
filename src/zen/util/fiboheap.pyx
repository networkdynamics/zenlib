cimport cfiboheap
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer
from python_ref cimport Py_INCREF, Py_DECREF 

DEF INTMIN = -2147483648

cdef inline object convert_fibheap_el_to_pycapsule(cfiboheap.fibheap_el* element):
	return PyCapsule_New(element, NULL, NULL)
	
cdef inline cfiboheap.fibheap_el* convert_pycapsule_to_fibheap_el(object element):
	return <cfiboheap.fibheap_el*>PyCapsule_GetPointer(element, NULL) 

cdef class FiboHeap:
	"""
	This is an implementation of a Fibonacci Heap. A Fibonacci Heap is
	a very efficient heap. The cost of an insert is O(1), and the amortized
	cost of an extract minimum is O(lgn). You can extract an already inserted
	item out of order in O(lgn). The way the Fibonacci heap obtains this is
	by delaying the organizing of the items until you extract.

	This code was modified and than ported to python with kind permission from:
	- John-Mark Gurney (http://resnet.uoregon.edu/~gurney_j/jmpc/fib.html)
	"""
	
	def __cinit__(FiboHeap self):
		"""
		Initialize the Fibonacci Heap.
		Calls function which makes a Fibonacci heap that does ordering based on an
		double key that is given in addition to the data.
		This menthod was modified in the original code so that you can eliminate the need to call
		a comparision function to order the data in the heap.
		"""
		self.treeptr = cfiboheap.fh_makekeyheap()
		if self.treeptr is NULL:
			raise MemoryError()
	
	def __dealloc__(FiboHeap self):
		if self.treeptr is not NULL:
			cfiboheap.fh_deleteheap(self.treeptr)

	cpdef object insert(FiboHeap self, double key, object data=None):
		"""
		Insert value into Fibonacci Heap.
		Function creates the data element based on provided values and insers into the heap.
		The pointer returned can be used in calls to functions like delete and decrease key.
		"""
		Py_INCREF(data)
		cdef cfiboheap.fibheap_el* retValue = cfiboheap.fh_insertkey(self.treeptr, key, <void*>data)
		if retValue is NULL:
			raise MemoryError()
		
		return convert_fibheap_el_to_pycapsule(retValue) 

	cdef void* _insert(FiboHeap self, double key, void* data=NULL):
		"""
		C Optimized version of the insert function.
		Function creates the data element based on provided values and insers into the heap.
		The pointer returned can be used in calls to functions like delete and decrease key.
		"""
		cdef cfiboheap.fibheap_el* ret = cfiboheap.fh_insertkey(self.treeptr, key, data)
		if ret is NULL:
			raise MemoryError()
		
		return <void*> ret
		
	cpdef object peek(FiboHeap self):
		"""
		Peek at Min Value on the Fibonacci Heap.
		Function returns the min element structure of the data at the top of the Fibonacci Heap without extracting it.
		"""
		cdef void* ret = cfiboheap.fh_min(self.treeptr)
		if ret is NULL:
			raise IndexError("FiboHeap is empty")
		
		return <object> ret
	
	cdef void * _peek(FiboHeap self):
		"""
		C Optimized version of the peek function.
		Function returns the min element structure of the data at the top of the Fibonacci Heap without extracting it.
		"""
		cdef void* ret = cfiboheap.fh_min(self.treeptr)
		if ret is NULL:
			raise IndexError("FiboHeap is empty")
		
		return <void*> ret
		
		
	cpdef object extract(FiboHeap self):
		"""
		Extract the Min Element from the Fibonacci Heap.
		Functions returns the heap min element structure, and deletes it from the heap.
		"""
		cdef void* ret = cfiboheap.fh_extractmin(self.treeptr)
		if ret is NULL:
			raise IndexError("FiboHeap is empty")
		
		return <object> ret 
		
	cdef void * _extract(FiboHeap self):
		"""
		C Optimized version of the extract function.
		Functions returns the heap min element structure, and deletes it from the heap.
		"""
		cdef void* ret = cfiboheap.fh_extractmin(self.treeptr)
		if ret is NULL:
			raise IndexError("FiboHeap is empty")
		
		return <void*> ret
			
	cpdef double get_min_key(FiboHeap self):
		"""
		Get the Min Key on the Fibonacci Heap.
		Function returns the integer key of the data at the top of the Fibonacci Heap without extracting it.
		"""
		cdef double ret = cfiboheap.fh_minkey(self.treeptr)
		if ret == INTMIN:
			raise IndexError("FiboHeap is empty")
			
		return ret
			
	cdef double _get_min_key(FiboHeap self):
		"""
		C Optimized version of the get_Min_Key function.
		Function returns the integer key of the data at the top of the Fibonacci Heap without extracting it.
		"""
		cdef double ret = cfiboheap.fh_minkey(self.treeptr)
		if ret == INTMIN:
			raise IndexError("FiboHeap is empty")
		
		return ret
	
	cpdef object decrease_key(FiboHeap self,  object element, double newKey):
		"""
		C Optimized version of the Deacrease Key function.
		function decrease the key in passed element with given key.
		"""
		cdef void* ret = cfiboheap.fh_replacekey(self.treeptr, convert_pycapsule_to_fibheap_el(element), newKey)
		if ret is NULL:
			raise IndexError("New Key is Bigger")
		
		return <object> ret
	
	cdef void* _decrease_key(FiboHeap self, void* element, double newKey):
		"""
		Decrease element key.
		function decrease the key in passed element with given key.
		"""
		cdef void* ret = cfiboheap.fh_replacekey(self.treeptr, <cfiboheap.fibheap_el*>element, newKey)
		if ret is NULL:
			raise IndexError("New Key is Bigger")
		
		return ret
	
	cpdef object replace_data(FiboHeap self, object element, object data):
		"""
		Replace Data in element.
		function replaces the Data in passed element with given data.
		"""
		cdef void* ret = cfiboheap.fh_replacedata(self.treeptr, convert_pycapsule_to_fibheap_el(element), <void*> data)
		if ret is NULL:
			raise IndexError()
			
		return <object> ret	
			
	cdef void* _replace_data(FiboHeap self, cfiboheap.fibheap_el* element, void* data):
		"""
		C Optimized version of the Replace Data  function.
		Function returns the integer key of the data at the top of the Fibonacci Heap without extracting it.
		"""
		cdef void* ret = cfiboheap.fh_replacedata(self.treeptr, element, data)
		if ret is NULL:
			raise IndexError()
			
		return ret

	cpdef object delete(FiboHeap self, object element):
		"""
		Delete the Min Value from the Fibonacci Heap.
		function removes from the Fibonacci heap, and returns the data that was stored in the element.
		"""
		cdef void* ret = cfiboheap.fh_delete(self.treeptr, convert_pycapsule_to_fibheap_el(element))
		if ret is NULL:
			raise MemoryError()
		
		return <object> ret
		
	cdef void* _delete(FiboHeap self, cfiboheap.fibheap_el* element):
		"""
		C Optimized version of the Delete Value from the Fibonacci Heap.
		function removes from the Fibonacci heap, and returns the data that was stored in the element.
		"""
		cdef void* ret = cfiboheap.fh_delete(self.treeptr, <cfiboheap.fibheap_el*>element)
		if ret is NULL:
			raise MemoryError()
			
		return ret

	cpdef heap_union(FiboHeap self, object heap):
		"""
		Union two Fibonacci Heap.
		function unites the current object with the passed heap.
		"""
		cfiboheap.fh_union(self.treeptr, <cfiboheap.fibheap*>heap)
	
	cdef _heap_union(FiboHeap self, cfiboheap.fibheap* heap):
		"""
		C Optimized version of the Union function.
		function unites the current object with the passed heap.
		"""
		cfiboheap.fh_union(self.treeptr, heap)
	
	cpdef int get_node_count(FiboHeap self):
		"""
		Return the Number of Nodes on the Fibonacci Heap.
		"""
		cdef int ret = cfiboheap.get_node_count(self.treeptr)
		if ret == INTMIN:
			raise MemoryError("Heap Not Initilaized")
		return cfiboheap.get_node_count(self.treeptr)
		
	cdef int _get_node_count(FiboHeap self):
		"""
		C Optimized Return the Number of Nodes on the Fibonacci Heap.
		"""
		cdef int ret = cfiboheap.get_node_count(self.treeptr)
		if ret == INTMIN:
			raise MemoryError("Heap Not Initilaized")
		return cfiboheap.get_node_count(self.treeptr)
		
	
	cpdef object get_node_data(FiboHeap self, object element):
		"""
		Return the given Node Data
		"""
		cdef void* ret = cfiboheap.get_fibheap_el_data(convert_pycapsule_to_fibheap_el(element))
		if ret is NULL:
			raise IndexError()
		return <object> ret
		
	cdef void* _get_node_data(FiboHeap self, cfiboheap.fibheap_el* element):
		"""
		C Optimized Return the given Node Data
		"""
		cdef void* ret = cfiboheap.get_fibheap_el_data(element)
		if ret is NULL:
			raise IndexError()	
		return ret

	cpdef double get_node_key(FiboHeap self, object element):
		"""
		Return the given Node Key
		"""
		cdef double ret = cfiboheap.get_fibheap_el_key(convert_pycapsule_to_fibheap_el(element))
		if ret == INTMIN:
			raise IndexError()
		return ret
		
	cdef double _get_node_key(FiboHeap self, cfiboheap.fibheap_el* element):
		"""
		C Optimized Return the given Node Key
		"""
		cdef double ret = cfiboheap.get_fibheap_el_key(element)
		if ret == INTMIN:
			raise IndexError()
		return ret
		