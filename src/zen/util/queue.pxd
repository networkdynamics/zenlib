from zen.digraph cimport *

cdef struct node:
	int nodeid
	NodeInfo nodedata
	
cdef union itemcontents:
	int intdata
	char* chardata
	node node

cdef struct queue_item:
	float priority
	itemcontents contents

cdef struct queue:
	int n
	queue_item* queue
	int space
	
cdef inline queue_create(queue* self,int initial_size)
cdef inline queue_destroy(queue* self)
cdef inline queue_resize(queue* self, int new_space)

cdef inline priority_queue_push(queue* self, queue_item item)
cdef inline queue_item priority_queue_peek(queue* self)
cdef priority_queue_remove(queue* self)
cdef inline queue_item priority_queue_pop(queue* self)


cdef inline queue_push(queue* self, queue_item item)
cdef inline queue_item queue_pop(queue* self)

	


