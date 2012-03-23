import unittest
import random
import types
import random

from zen import *

class HGraphTestCase(unittest.TestCase):
	
	def test_tograph1(self):
		H = HyperGraph()
		H.add_edge([1,2,3])
		H.add_edge([2,4])
		
		G = H.to_graph()
		
		self.assertTrue(G.has_edge(1,2))
		self.assertTrue(G.has_edge(1,3))
		self.assertTrue(G.has_edge(2,3))
		self.assertTrue(G.has_edge(2,4))
		
		self.assertFalse(G.has_edge(1,4))
		self.assertFalse(G.has_edge(3,4))
	
	def test_basic1(self):
		H = HyperGraph()
		n0 = H.add_node(1)
		n1 = H.add_node(2)
		n2 = H.add_node(3)
		n3 = H.add_node(4)
		
		e1 = H.add_edge([1,2,3,4])
		e2 = H.add_edge([2,3,4,5])
		
		# test endpoints
		self.assertEqual(set(H.endpoints_(e1)),set([n0,n1,n2,n3]))
		
		# test nodes
		self.assertEqual(set(H.nodes()),set([1,2,3,4,5]))
		self.assertEqual(set(H.nodes_iter()),set([1,2,3,4,5]))
		
		# test edge containment
		self.assertTrue(H.has_edge([1,2,3,4]))
		self.assertFalse(H.has_edge([1,2,3]))
		self.assertEqual(H.edge_idx([2,3,4,5]),e2)
		
		# test neighbors
		self.assertEqual(set(H.neighbors(1)),set([2,3,4]))
		self.assertTrue(H.is_neighbor(2,4))
		self.assertFalse(H.is_neighbor(5,1))
		self.assertFalse(H.is_neighbor(1,5))
		
	def test_degree_card(self):
		H = HyperGraph()
		e1 = H.add_edge([1,2,3,4])
		e2 = H.add_edge([2,3,4,5,6])
		
		self.assertEqual(H.degree(1),1)
		self.assertEqual(H.degree(2),2)
		
		self.assertEqual(H.card_(e1),4)
		self.assertEqual(H.card_(e2),5)
		
	def test_grow_elist(self):
		H = HyperGraph(edge_list_capacity=5)
		for i in range(1,8):
			H.add_edge([0,i])
		
	def test_add_lots_of_nodes(self):
		H = HyperGraph()
		for i in range(1000):
			H.add_node(i)
		
	def test_add_lots_of_edges(self):
		nodes = range(500)
		H = HyperGraph()
		
		for i in range(1000):
			H.add_edge(random.sample(nodes,5))