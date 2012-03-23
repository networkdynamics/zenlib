"""Fibonacci Heap
   
   Fibonacci heap is a heap data structure consisting of a collection of trees.
   Find-minimum is O(log(n)) amortized time. Operations insert, decrease key, and merge (union) work in constant amortized time. 
   Operations delete and delete minimum work in O(log n) amortized time.
   A Fibonacci heap is thus better than a binomial heap when b is asymptotically smaller than a.
   Using Fibonacci heaps for priority queues improves the asymptotic running time of important algorithms, 
   such as Dijkstra's algorithm for computing Shortest paths.

   This heap was implemented based on:
   Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. Introduction to Algorithms, Second Edition. 
   MIT Press and McGraw-Hill, 2001. ISBN 0-262-03293-7. Chapter 20: Fibonacci Heaps, pp.476–497. Third edition p518.
"""
cimport libc.stdlib
import numpy as np
cimport numpy as np

cdef extern from "math.h":
	float log(float theta)
	float ceil(float theta)

cdef class FibNode:
	 
	def __cinit__(FibNode self, key, data=None):
		self.key = key
		#self.data = None
		self.data = data
		self.reset()
	
	def reset(FibNode self):
		self.degree = 0
		self.parent = None
		self.child = None
		self.left = self
		self.right = self
		self.mark = False
	
	cpdef get_key(FibNode self):
		return self.key

	cpdef print_node(FibNode self, indent='#', charr='#'):
		cdef FibNode x
		x = self
		first_iteration = True
		while x != None and (first_iteration or x != self):
			first_iteration = False
			print "%s %s [%d]" %(indent, x.key, x.degree)
			if x.child != None:
				x.child.print_node(indent+charr, charr)
			x = x.right
	
	cpdef set_data(FibNode self, object data):
		self.data = data
	
	cpdef object get_data(FibNode self):
		return self.data

	cpdef FibNode get_left(FibNode self):
		return self.left

	cpdef FibNode get_right(FibNode self):
		return self.right

	cpdef FibNode get_parent(FibNode self):
		return self.parent

	cpdef FibNode get_child(FibNode self):
		return self.child

	cdef __append_nodes(FibNode self, FibNode nodeX):
		"""Add list y to x."""
		cdef FibNode tempNode

		if self == None or nodeX == None:
			return
		tempNode = self
		while tempNode.right != self:
			tempNode.parent = nodeX.parent
			tempNode = tempNode.right
		tempNode.parent = nodeX.parent
		self.left = nodeX.left
		nodeX.left.right = self
		tempNode.right = nodeX
		nodeX.left = tempNode

	cdef __link(FibNode self, FibNode nodeX):
		self.left.right = self.right
		self.right.left = self.left
		self.parent = nodeX
		if nodeX.child != None:
			nodeX.child.left.right = self
			self.left = nodeX.child.left
			self.right = nodeX.child
			nodeX.child.left = self
		else:
			nodeX.child = self
			self.left = self
			self.right = self
		nodeX.degree = nodeX.degree + 1
		self.mark = False

	cdef __print_node_info(FibNode self):
	
		print """
		--- node info ---
		key = %.1f
		""" % (self.key)

	# add set method

cdef class FibHeap:
	"""
	
	"""
	
	def __cinit__(FibHeap self):
		self.min = None
		self.nodeCount = 0
   	
	cpdef FibNode get_min(FibHeap self):
		return self.min
	
	def get_node_count(FibHeap self):
		return self.nodeCount

	def is_empty(FibHeap self):
		return self.nodeCount == 0
	
	cpdef get_all_nodes(FibHeap self):
		
		nodes = np.empty((self.nodeCount), dtype=np.object_)
		# TODO

		return nodes
	
	cpdef insert_node(FibHeap self, FibNode node):
		cdef FibNode tempMinLeft
		cdef FibNode tempMin
		node.reset()
		if self.min == None:
			self.min = node
		else:
			tempMinLeft = self.min.left
			tempMin = self.min
			node.left = tempMinLeft
			node.right = tempMin
			tempMinLeft.right = node
			tempMin.left = node
			if node.key < self.min.key:
				self.min = node
		self.nodeCount = self.nodeCount + 1

	cpdef FibNode extract_node(FibHeap self):
		cdef FibNode node
		node = self.min
		if node != None:
			if node.child != None:
				node.child.__append_nodes(node)
			node.left.right = node.right
			node.right.left = node.left
			if node == node.right:
				self.min = None
			else:
				self.min = node.right
				self.__consolidate()
			self.nodeCount = self.nodeCount - 1
			node.reset()
		return node

	cpdef object extract_min_node_data(FibHeap self):
		cdef FibNode node
		node = self.extract_node()
		return node.data

	cpdef decrease_key(FibHeap self, FibNode node, double key):
		cdef FibNode tempParent

		assert(key <= node.key)
		if key == node.key:
			return
		node.key = key
		tempParent = node.parent
		if  tempParent != None and node.key < tempParent.key:
			self.__cut(node, tempParent)
			self.__cascading_cut(tempParent)
		if node.key < self.min.key:
			self.min = node

	cpdef show_heap(FibHeap self):
		self.min.print_node('o', '->')	

	cdef __consolidate(FibHeap self):
		cdef FibNode x
		cdef FibNode y
		cdef unsigned int max_degree = <unsigned int>ceil(log(self.nodeCount))
		cdef unsigned int d
		
		A = []
		for i from 0 <= i <= max_degree*2:
			A.append(None)
		
		root_list = []
		x = self.min
		x.left.right = None
		while x != None:
			next_x = x.right
			x.left = x
			x.right = x
			root_list.append(x)
			x = next_x
		for x in root_list:
			#x.__print_node_info()
			d = x.degree
			#print "index=", d, "max_degree=", max_degree, "heap.nodeCount=",heap.nodeCount
			while A[d] != None:
				y = A[d]
				if y.key < x.key:
					x,y = y,x
				y.__link(x)
				A[d] = None
				d = d + 1
			A[d] = x
		self.min = None
		for x in A:
			if x != None:
				x.left = x
				x.right = x
				x.parent = None
				if self.min == None:
					self.min = x
				else:
					x.__append_nodes(self.min)
					if x.key < self.min.key:
						self.min = x

	cdef __cut(FibHeap self, FibNode nodeX, FibNode nodeY):
		# remove x from the child list of y, decrementing y.degree.
		nodeX.left.right = nodeX.right
		nodeX.right.left = nodeX.left
		nodeY.degree = nodeY.degree - 1
		nodeY.child = nodeX.right
		if nodeX == nodeX.right:
			nodeY.child = None
		# add x to the root list of H
		nodeX.parent = None
		nodeX.mark = False
		nodeX.left = self.min.left
		nodeX.right = self.min
		nodeX.left.right = nodeX
		nodeX.right.left = nodeX
		#x.__append_nodes(heap.min)

	cdef __cascading_cut(FibHeap self, FibNode node):
		cdef FibNode parentNode
		parentNode = node.parent
		if parentNode != None:
			if node.mark == False:
				node.mark = True
			else:
				self.__cut(node, parentNode)
				self.__cascading_cut(parentNode)

	cdef FibHeap __union(FibHeap self, FibHeap heap2):
		heap = FibHeap()
		if self.min != None and heap2.min == None:
			heap.min = self.min
			heap.nodeCount = self.nodeCount
		elif self.min == None and heap2.min != None:
			heap.min = heap2.min
			heap.nodeCount = heap2.nodeCount
		elif self.min != None and heap2.min != None:
			heap2.min.__append_nodes(self.min)
			if self.min.key <= heap2.min.key:
				heap.min = self.min
			else:
				heap.min = heap2.min
			heap.nodeCount = self.nodeCount + heap2.nodeCount
		return heap
