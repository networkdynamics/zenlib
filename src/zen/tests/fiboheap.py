#!/usr/bin/env python
# encoding: utf-8
"""
heap.py

"""
import unittest
from zen import *
from zen.util.fiboheap import FiboHeap
import random
import time
import gc
from heapq import heappush, heappop

class TestZenFibonacciHeap(unittest.TestCase):
	
	def test_insert_data(self):
		minNode=None
	
		H = FiboHeap()
		random.seed()
		
		for d in range(10000):
			n = random.randint(1, 100000)
			if minNode == None or n < minNode:
				minNode = n
			
			H.insert(n,n)
		
		self.assertEqual(H.get_min_key(), minNode)
	
	def test_insert_extract_order(self):
		
		nodes = []
		extractedNodes = []
		
		H = FiboHeap()
		random.seed()
		
		for d in range(10000):
			n = random.randint(1, 100000)
			nodes.append(n)
			H.insert(n,n)

		for i in range(10000):
			n = H.get_min_key()
			H.extract()
			extractedNodes.append(n)
		
		nodes.sort()
	
		self.assertEqual(extractedNodes, nodes)
	
	def test_decrease_key(self):
			nodes = []
			maxKey = None
			maxNode = None
			
		
			H = FiboHeap()
			random.seed()
			
			for d in range(10000):
				n = random.randint(1, 100000)
				if maxNode == None or n > maxNode:
					maxKey = n
					maxNode = H.insert(n,n)
					nodes.append(n)
				else: 
					nodes.append(n)
					H.insert(n,n)
			
			H.decrease_key(maxNode, H.get_min_key())	
			
			self.assertEqual(H.get_node_key(maxNode), H.get_min_key())
	 		
	def test_rigorous_insert_extract(self):
		
		H = FiboHeap()
		random.seed()
		
		for d in range(100000):
			n = random.randint(1, 100000)
			m = H.insert(n,n)
			H.decrease_key(m, n-1)
			H.extract()
			
	#This test is not possible in the new fibheap need to check with Derek alternatives 
	#def test_left_right_refs(self):
	# 		
	# 		# the heap will look like a linked list i.e. 10 - 11 - 10 - 12 at the end of the test
	# 		# this is testing the references between the nodes
	# 
	# 		H = FibHeap()
	# 		
	# 		n1 = FibNode(10)
	# 		n2 = FibNode(11)
	# 		n3 = FibNode(10)
	# 		n4 = FibNode(12)
	# 		
	# 		H.insert_node(n1)
	# 		H.insert_node(n2)
	# 		H.insert_node(n3)
	# 		H.insert_node(n4)
	# 
	# 		self.assertEquals(n4, n1.get_left())
	# 		self.assertEquals(n1, n4.get_right())
	# 		self.assertEquals(n2, n1.get_right())
	# 		self.assertEquals(n1, n2.get_left())
	# 		self.assertEquals(n3, n2.get_right())
	# 		self.assertEquals(n2, n3.get_left())
	# 		self.assertEquals(n4, n3.get_right())
	# 		self.assertEquals(n3, n4.get_left())
	
	#This test is not possible in the new fibheap need to check with Derek alternatives 	
	# def test_child_ref(self):
	# 
	# 		H = FibHeap()
	# 
	# 		n1 = FibNode(1)
	# 		n2 = FibNode(2)
	# 		n3 = FibNode(3)
	# 
	# 		H.insert_node(n1)
	# 		H.insert_node(n2)
	# 		H.insert_node(n3)
	# 
	# 		self.assertIsNone(n1.get_child())
	# 		self.assertIsNone(n2.get_child())
	# 		self.assertIsNone(n3.get_child())
	#         
	# 		H.extract_node()
	# 
	# 		self.assertEquals(n3, n2.get_child())
	# 		self.assertIsNone(n3.get_child())
		
	def test_object_data(self):
		
		H = FiboHeap()

		n1 = H.insert(1, 'this is a string')
		
		self.assertEquals('this is a string', H.get_node_data(n1))

		n2 = H.insert(2)
		H.replace_data(n2,(1,2))
		self.assertEquals((1,2), H.get_node_data(n2))

		self.assertEquals('this is a string', H.extract())
	
	def test_speed_compare(self):
	
		H = FiboHeap()
		random.seed()

		n = 10000
		
		gc.disable()
		t1 = time.time()
		for d in range(n):
			H.insert(random.randint(1, 100000), "Random Data")
			
		for d in range(n):
			H.extract()
		
		t2 = time.time()
		zenFinal = t2-t1
		gc.collect()
		
		h = []
		t1 = time.time()
		for d in range(n):
			heappush(h, random.randint(1, 100000))

		for d in range(n):
			heappop(h)

		t2 = time.time()
		pythonFinal = t2-t1
		gc.enable()
		
		self.assertGreater(pythonFinal, zenFinal)
		
if __name__ == '__main__':
	unittest.main()
