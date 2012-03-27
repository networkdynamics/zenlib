import unittest

import zen

class UndirectedERTestCase(unittest.TestCase):
	
	def test_bad_argument(self):
		try:
			zen.generating.erdos_renyi(10,0.5,blah=10)
			self.fail('blah should not be accepted as a keyword argument')
		except zen.ZenException,e:
			pass
			
	def test_seed(self):
		G1 = zen.generating.erdos_renyi(10,0.5,seed=10)
		G2 = zen.generating.erdos_renyi(10,0.5,seed=10)
		
		for e in G1.edges_iter():
			if not G2.has_edge(*e):
				self.fail('Graphs generated using same seed are different.')
				
		for e in G2.edges_iter():
			if not G1.has_edge(*e):
				self.fail('Graphs generated using same seed are different.')
	
	def test_complete(self):
		G = zen.generating.erdos_renyi(5,1.0,self_loops=True)
		
		for i in G.nodes():
			for j in G.nodes():
				self.assertTrue(G.has_edge(i,j))

	
	def test_empty(self):
		G = zen.generating.erdos_renyi(5,0.0,self_loops=True)
		
		for i in G.nodes():
			for j in G.nodes():
				self.assertFalse(G.has_edge(i,j))
				
class DirectedERTestCase(unittest.TestCase):

	def test_complete(self):
		G = zen.generating.erdos_renyi(5,1.0,directed=True,self_loops=True)

		for i in G.nodes():
			for j in G.nodes():
				self.assertTrue(G.has_edge(i,j))


	def test_empty(self):
		G = zen.generating.erdos_renyi(5,0.0,directed=True,self_loops=True)

		for i in G.nodes():
			for j in G.nodes():
				self.assertFalse(G.has_edge(i,j))