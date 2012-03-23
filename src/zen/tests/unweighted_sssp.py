import unittest
from zen import *

class UWSSSPTestCase(unittest.TestCase):
	
	def test_sssp_undirected(self):
        
		# following example from CLRS book page 596

		G = Graph()
		G.add_edge('o', 'a')
		G.add_edge('a', 'f')
		G.add_edge('f', 't')
		G.add_edge('t', 'e')
		G.add_edge('e', 'c')
		G.add_edge('c', 'o')
		G.add_edge('o', 'b')
		G.add_edge('a', 'b')
		G.add_edge('a', 'd')
		G.add_edge('b', 'd')
		G.add_edge('b', 'e')
		G.add_edge('t', 'd')
		G.add_edge('e', 'd')
		G.add_edge('b', 'c')
		G.add_edge('x', 'y')

		D = single_source_shortest_path(G, 'o')

		self.assertEqual(D['o'], (0, None))
		self.assertEqual(D['a'][0], 1)
		self.assertEqual(D['b'][0], 1)
		self.assertEqual(D['d'][0], 2)
        
		self.assertFalse('x' in D) 
		self.assertFalse('y' in D) 

	def test_unreachable(self):
        
		G = DiGraph()
		G.add_edge('s', 't', None, 10)
		G.add_edge('x', 'z', None, 2)

		d,p = single_source_shortest_path(G, 's', 'x')
		
		self.assertEquals(None, d)
		self.assertEquals(None, p)

	def test_sssp_directed(self):
        
		# following example from CLRS book page 596

		G = DiGraph()
		G.add_edge('s', 't')
		G.add_edge('s', 'y')
		G.add_edge('t', 'x')
		G.add_edge('t', 'y')
		G.add_edge('y', 't')
		G.add_edge('y', 'x')
		G.add_edge('y', 'z')
		G.add_edge('z', 's')
		G.add_edge('z', 'x')
		G.add_edge('x', 'z')
		G.add_edge('a', 'b')

		D = single_source_shortest_path(G, 's')

		self.assertEqual(D['s'], (0, None))
		self.assertEqual(D['y'], (1, 's'))
        
		self.assertFalse('a' in D) 
		self.assertFalse('b' in D) 

	def test_spsp_directed(self):
		# following example from CLRS book page 596

		G = DiGraph()
		G.add_edge('s', 'y')
		G.add_edge('t', 'x')
		G.add_edge('t', 'y')
		G.add_edge('y', 't')
		G.add_edge('y', 'x')
		G.add_edge('y', 'z')
		G.add_edge('z', 's')
		G.add_edge('z', 'x')
		G.add_edge('x', 'z')
		G.add_edge('a', 'b')

		d,p = single_source_shortest_path(G, 's', 't') # x should not be found
       	
		self.assertEquals(2, d)
		
		self.assertEquals(['s', 'y', 't'], p)

if __name__ == '__main__':
	unittest.main()
