import unittest
from zen import *
import networkx
import random

class DijkstraTestCase(unittest.TestCase):
	
	def test_sssp_undirected(self):
        
		# following example from CLRS book page 596

		G = Graph()
		G.add_edge('o', 'a', None, 2)
		G.add_edge('a', 'f', None, 12)
		G.add_edge('f', 't', None, 3)
		G.add_edge('t', 'e', None, 7)
		G.add_edge('e', 'c', None, 4)
		G.add_edge('c', 'o', None, 4)
		G.add_edge('o', 'b', None, 5)
		G.add_edge('a', 'b', None, 2)
		G.add_edge('a', 'd', None, 7)
		G.add_edge('b', 'd', None, 4)
		G.add_edge('b', 'e', None, 3)
		G.add_edge('t', 'd', None, 5)
		G.add_edge('e', 'd', None, 1)
		G.add_edge('b', 'c', None, 1)
		G.add_edge('x', 'y', None, 1)

		D = dijkstra(G, 'o')

		self.assertEqual(D['o'], (0, None))
		self.assertEqual(D['a'], (2, 'o'))
		self.assertEqual(D['b'], (4, 'a'))
		#self.assertEqual(D['d'], (8, 'e'))
		self.assertEqual(D['c'], (4, 'o'))
		self.assertEqual(D['e'], (7, 'b'))
		self.assertEqual(D['t'], (13, 'd'))
		self.assertEqual(D['f'], (14, 'a'))
        
		self.assertFalse('x' in D) 
		self.assertFalse('y' in D) 

	def test_source_is_end(self):
        
		G = DiGraph()
		G.add_edge('s', 't', None, 10)

		d, p = dijkstra(G, 's', 's')
		
		self.assertEquals(0, d)
		self.assertEquals([], p)

	def test_unreachable(self):
        
		G = DiGraph()
		G.add_edge('s', 't', None, 10)
		G.add_edge('x', 'z', None, 2)

		d, p = dijkstra(G, 's', 'x')
		
		self.assertEquals(None, d)
		self.assertEquals(None, p)

	def test_sssp_directed(self):
        
		# following example from CLRS book page 596

		G = DiGraph()
		G.add_edge('s', 't', None, 10)
		G.add_edge('s', 'y', None, 5)
		G.add_edge('t', 'x', None, 1)
		G.add_edge('t', 'y', None, 2)
		G.add_edge('y', 't', None, 3)
		G.add_edge('y', 'x', None, 9)
		G.add_edge('y', 'z', None, 2)
		G.add_edge('z', 's', None, 7)
		G.add_edge('z', 'x', None, 6)
		G.add_edge('x', 'z', None, 4)
		G.add_edge('a', 'b', None, 4)

		D = dijkstra(G, 's')

		self.assertEqual(D['s'], (0, None))
		self.assertEqual(D['t'], (8, 'y'))
		self.assertEqual(D['y'], (5, 's'))
		self.assertEqual(D['x'], (9, 't'))
		self.assertEqual(D['z'], (7, 'y'))
        
		self.assertFalse('a' in D) 
		self.assertFalse('b' in D) 

	def test_spsp_directed(self):
		# following example from CLRS book page 596

		G = DiGraph()
		G.add_edge('s', 't', None, 10)
		G.add_edge('s', 'y', None, 5)
		G.add_edge('t', 'x', None, 1)
		G.add_edge('t', 'y', None, 2)
		G.add_edge('y', 't', None, 3)
		G.add_edge('y', 'x', None, 9)
		G.add_edge('y', 'z', None, 2)
		G.add_edge('z', 's', None, 7)
		G.add_edge('z', 'x', None, 6)
		G.add_edge('x', 'z', None, 4)
		G.add_edge('a', 'b', None, 4)

		d,p = dijkstra(G, 's', 't') # x should not be found
       	
		self.assertEquals(8, d)
		
		self.assertEquals(['s', 'y', 't'], p)
		
	def test_simple_directed_(self):
		G = Graph()
		G.add_edge(0,1)
		G.add_edge(1,2)
		G.add_edge(2,3)
		
		R = dijkstra_(G,0,3)
		
		self.assertEquals(len(R),2)
		d,p = R
		self.assertEquals(d[3],3)
		
		return

if __name__ == '__main__':
	unittest.main()
