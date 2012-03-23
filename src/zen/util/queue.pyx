# priority queue

from zen.digraph cimport *
cimport stdlib

__all__ = ['queue_create', 'queue_destroy','queue_resize', 'priority_queue_push', 'priority_queue_peek', 'priority_queue_remove', 'priority_queue_pop','queue_push','queue_pop' ]


cdef struct node:
	int nodeid
	NodeInfo nodedata
	
cdef union itemcontents:
	int intdata
	char* chardata
	node node

cdef struct queue_item:
	double priority
	itemcontents contents

cdef struct queue:
	int n
	queue_item* queue
	int space


cdef inline queue_create(queue* self,int initial_size):
	self.space = initial_size
	self.queue = <queue_item*>stdlib.malloc(sizeof(queue_item)*self.space)
	self.n=0

cdef inline queue_destroy(queue* self):
	stdlib.free(self.queue)

cdef inline queue_resize(queue* self, int new_space):
	if new_space<self.n:
		raise ValueError("queue attempt to resize to %d failed with %d items on queue" % (new_space, self.n))
	self.space = new_space
	self.queue = <queue_item*>stdlib.realloc(<void*>self.queue,new_space*sizeof(queue_item))

cdef inline priority_queue_push(queue* self, queue_item item):
	cdef int i
	cdef queue_item t

	self.n += 1
	if self.n>self.space:
		queue_resize(self,2*self.space+1)

	i = self.n-1
	self.queue[i] = item
	while i>0 and self.queue[i].priority<self.queue[(i-1)//2].priority:	
		t = self.queue[(i-1)//2]
		self.queue[(i-1)//2] = self.queue[i]
		self.queue[i] = t
		i = (i-1)//2

cdef queue_item priority_queue_peek(queue* self):
	return self.queue[0]

cdef priority_queue_remove(queue* self):
	cdef queue_item t
	cdef int i, j, k, l

	self.queue[0] = self.queue[self.n-1]
	self.n -= 1
	if self.n < self.space//4 and self.space>40:
		queue_resize(self,self.space//2+1)
	
	i=0
	j=1
	k=2
	while ((j<self.n and 
		self.queue[i].priority > self.queue[j].priority or
			k<self.n and 
				self.queue[i].priority > self.queue[k].priority)):
		if k<self.n and self.queue[j].priority>self.queue[k].priority:
			l = k
		else:
			l = j
			t = self.queue[l]
		self.queue[l] = self.queue[i]
		self.queue[i] = t
		i = l
		j = 2*i+1
		k = 2*i+2

cdef queue_item priority_queue_pop(queue* self):	
	cdef queue_item item
	item = priority_queue_peek(self)
	priority_queue_remove(self)
	return item

cdef inline queue_push(queue* self, queue_item item):
	cdef int i

	self.n += 1
	if self.n>self.space:
		queue_resize(self,2*self.space+1)

	i = self.n-1
	self.queue[i] = item
		
cdef inline queue_item queue_pop(queue* self):
	cdef queue_item item
	i = self.n-1
	item = self.queue[i] 
	
	self.n -= 1
	if self.n < self.space//4 and self.space>40:
		queue_resize(self,self.space//2+1)
	
	return item
	