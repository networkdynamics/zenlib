import unittest

import zen

class UndirectedERTestCase(unittest.TestCase):
	
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