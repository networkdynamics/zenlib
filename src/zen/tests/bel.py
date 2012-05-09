from zen import *
import unittest
import os.path as path
import random

class BELTestCase(unittest.TestCase):

	def test_undirected_not_compact(self):
		G = Graph()
		G.add_node(1)
		G.add_node(2)
		G.rm_node(1)
		
		try:
			x = bel.write_str(G)
			self.fail('bel.write_str should have raised an exception. G is not compact.')
		except ZenException,e:
			pass
	
	def test_write_objless(self):
		G = DiGraph()
		n1 = G.add_node()
		n2 = G.add_node()
		
		G.add_edge_(n1,n2)
		
		x = bel.write_str(G)
		G2 = bel.read_str(x,directed=True)
		
		self.assertTrue(G2.has_edge_(n1,n2))
			
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