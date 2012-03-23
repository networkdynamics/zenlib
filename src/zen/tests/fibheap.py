#!/usr/bin/env python
# encoding: utf-8
"""
heap.py

"""
import unittest
from zen import *
from zen.util.fibheap import FibHeap, FibNode
import random
import time
import gc
from heapq import heappush, heappop

class TestZenFibonacciHeap(unittest.TestCase):
	
	def test_insert_data(self):
		nodes = []
		minNode=None
	
		H = FibHeap()
		random.seed()
		
		for d in range(10000):
			n = FibNode(random.randint(1, 100000))
			if minNode == None or n.get_key() < minNode.get_key():
				minNode = n
			nodes.append(n)
			H.insert_node(n)
		
		self.assertEqual(H.get_min(), minNode)
	
	def test_insert_extract_order(self):
		
		nodes = []
		extractedNodes = []
		
		H = FibHeap()
		random.seed()
		
		for d in range(10000):
			n = FibNode(random.randint(1, 100000))
			nodes.append(n.get_key())
			H.insert_node(n)

		for i in range(H.get_node_count()):
			n = H.extract_node()
			extractedNodes.append(n.get_key())
		
		nodes.sort()
	
		self.assertEqual(extractedNodes, nodes)
	
	def test_decrease_key(self):
		nodes = []
		maxNode = None
	
		H = FibHeap()
		random.seed()
		
		for d in range(10000):
			n = FibNode(random.randint(1, 100000))
			if maxNode == None or n.get_key() > maxNode.get_key():
				maxNode = n
			nodes.append(n)
			H.insert_node(n)
		
		H.decrease_key(maxNode, H.get_min().get_key())	
		
		self.assertEqual(maxNode.get_key(), H.get_min().get_key())
		
	def test_rigorous_insert_extract(self):
		
		H = FibHeap()
		random.seed()
		
		for d in range(10000):
			n = FibNode(random.randint(1, 100000))
			H.insert_node(n)
			H.extract_node()
	
	def test_left_right_refs(self):
		
		# the heap will look like a linked list i.e. 10 - 11 - 10 - 12 at the end of the test
		# this is testing the references between the nodes

		H = FibHeap()
		
		n1 = FibNode(10)
		n2 = FibNode(11)
		n3 = FibNode(10)
		n4 = FibNode(12)
		
		H.insert_node(n1)
		H.insert_node(n2)
		H.insert_node(n3)
		H.insert_node(n4)

		self.assertEquals(n4, n1.get_left())
		self.assertEquals(n1, n4.get_right())
		self.assertEquals(n2, n1.get_right())
		self.assertEquals(n1, n2.get_left())
		self.assertEquals(n3, n2.get_right())
		self.assertEquals(n2, n3.get_left())
		self.assertEquals(n4, n3.get_right())
		self.assertEquals(n3, n4.get_left())
	
	def test_child_ref(self):

		H = FibHeap()

		n1 = FibNode(1)
		n2 = FibNode(2)
		n3 = FibNode(3)

		H.insert_node(n1)
		H.insert_node(n2)
		H.insert_node(n3)

		self.assertIsNone(n1.get_child())
		self.assertIsNone(n2.get_child())
		self.assertIsNone(n3.get_child())
        
		H.extract_node()

		self.assertEquals(n3, n2.get_child())
		self.assertIsNone(n3.get_child())
	
	def test_object_data(self):
		
		H = FibHeap()

		n1 = FibNode(1, 'this is a string')
		
		self.assertEquals('this is a string', n1.get_data())

		n2 = FibNode(2)
		n2.set_data((1,2))
		self.assertEquals((1,2), n2.get_data())

		H.insert_node(n1)
		H.insert_node(n2)
		
		self.assertEquals('this is a string', H.extract_min_node_data())
	
	def test_speed_compare(self):
		
		H = FibHeap()
		random.seed()

		n = 10000
		
		gc.disable()
		t1 = time.time()
		for d in range(n):
			H.insert_node(FibNode(random.randint(1, 100000)))
			
		for d in range(n):
			H.extract_node()
		
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
