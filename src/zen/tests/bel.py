from zen import *
import unittest
import os.path as path
import random

class BELTestCase(unittest.TestCase):

	def test_directed_rw1(self):
		G = DiGraph()
		x = range(100)
		for n in x:
			G.add_node(n)
			
		random.shuffle(x)
		
		G.add_edge(1,2)
		
		while len(x) > 0:
			a = x.pop()
			b = x.pop()
			
			if not G.has_edge(a,b):
				G.add_edge(a,b)
				
		G.compact()
		Gencoding = bel.write_str(G)
		G2 = bel.read_str(Gencoding,directed=True)
		
		self.assertEqual(type(G),type(G2))
		self.assertEqual(len(G),len(G2))
		self.assertEqual(G.size(),G2.size())
		
		x = range(100)
		for a in x:
			for b in x:
				self.assertEqual(G.has_edge_(a,b),G2.has_edge_(a,b))

	def test_undirected_rw1(self):
		G = Graph()
		x = range(100)
		for n in x:
			G.add_node(n)

		random.shuffle(x)
		while len(x) > 0:
			a = x.pop()
			b = x.pop()

			if not G.has_edge(a,b):
				G.add_edge(a,b)
		
		G.compact()
		Gencoding = bel.write_str(G)
		G2 = bel.read_str(Gencoding)
		
		self.assertEqual(type(G),type(G2))
		self.assertEqual(len(G),len(G2))
		self.assertEqual(G.size(),G2.size())

		x = range(100)
		for a in x:
			for b in x:
				self.assertEqual(G.has_edge_(a,b),G2.has_edge_(a,b))
	
if __name__ == '__main__':
	unittest.main()