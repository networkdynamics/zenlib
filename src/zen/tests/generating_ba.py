import unittest

import zen

class UndirectedBATestCase(unittest.TestCase):
	
	def test_seed(self):
		G1 = zen.generating.barabasi_albert(10,3,seed=10)
		G2 = zen.generating.barabasi_albert(10,3,seed=10)
		
		for e in G1.edges_iter():
			if not G2.has_edge(*e):
				self.fail('Graphs generated using same seed are different.')
				
		for e in G2.edges_iter():
			if not G1.has_edge(*e):
				self.fail('Graphs generated using same seed are different.')
	
	def test_undirected(self):
		G1 = zen.generating.barabasi_albert(10,3,seed=10)

		self.assertEqual(type(G1),zen.Graph)
		self.assertEqual(len(G1),10)
		
class DirectedBATestCase(unittest.TestCase):
	
	def test_directed(self):
		G1 = zen.generating.barabasi_albert(10,3,seed=10,directed=True)

		self.assertEqual(type(G1),zen.DiGraph)
		self.assertEqual(len(G1),10)