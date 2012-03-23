import unittest
from zen import *

class FloydWarshallTestCase(unittest.TestCase):
	
	def test_simple_directed(self):
		
		G = DiGraph()
		G.add_edge('x','y')
		G.add_edge('y','z')
		G.add_edge('z','x')
		
		D = floyd_warshall(G)
		
		self.assertEqual(D[('x','y')],1)
		self.assertEqual(D[('x','z')],2)
		self.assertEqual(D[('z','x')],1)
		self.assertEqual(D[('x','x')],0)
		
	def test_simple_undirected(self):

		G = Graph()
		G.add_edge('x','y')
		G.add_edge('y','z')
		G.add_edge('z','x')

		D = floyd_warshall(G)

		self.assertEqual(D[('x','y')],1)
		self.assertEqual(D[('x','z')],1)
		self.assertEqual(D[('z','x')],1)
		self.assertEqual(D[('x','x')],0)