import unittest
from zen import *
import networkx
import random

class AllPairsBellmanFordPathLength_TestCase(unittest.TestCase):
	
	def test_apdp_undirected(self):
		G = Graph()
		
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(2,4)

		D = all_pairs_bellman_ford_path_length_(G)
		
		self.assertEqual(D[0,0],0)
		self.assertEqual(D[0,1],1)
		self.assertEqual(D[0,2],2)
		self.assertEqual(D[0,3],2)
		self.assertEqual(D[1,2],1)
		self.assertEqual(D[1,3],1)
		self.assertEqual(D[2,3],2)
		
	def test_apdp_directed(self):
		G = DiGraph()

		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(2,4)

		D = all_pairs_bellman_ford_path_length_(G)

		self.assertEqual(D[0,0],0)
		self.assertEqual(D[0,1],1)
		self.assertEqual(D[0,2],2)
		self.assertEqual(D[0,3],2)
		self.assertEqual(D[1,2],1)
		self.assertEqual(D[1,3],1)
		self.assertEqual(D[2,3],float('infinity'))
		
	def test_disconnected(self):
		G = Graph()
		
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_node(4)

		D = all_pairs_bellman_ford_path_length_(G)
		
		self.assertEqual(D[0,3],float('infinity'))
		self.assertEqual(D[1,3],float('infinity'))
		self.assertEqual(D[2,3],float('infinity'))

class AllPairsBellmanFordPath_TestCase(unittest.TestCase):
	
	def test_apdp_undirected(self):
		G = Graph()
		
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(2,4)

		D,P = all_pairs_bellman_ford_path_(G)
		
		self.assertEqual(D[0,0],0)
		self.assertEqual(D[0,1],1)
		self.assertEqual(D[0,2],2)
		self.assertEqual(D[0,3],2)
		self.assertEqual(D[1,2],1)
		self.assertEqual(D[1,3],1)
		self.assertEqual(D[2,3],2)
		
		self.assertEqual(P[0,0],-1)
		self.assertEqual(P[0,1],0)
		self.assertEqual(P[0,2],1)
		self.assertEqual(P[0,3],1)
		self.assertEqual(P[1,2],1)
		self.assertEqual(P[1,3],1)
		self.assertEqual(P[2,3],1)
		
	def test_apdp_directed(self):
		G = DiGraph()

		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(2,4)

		D,P = all_pairs_bellman_ford_path_(G)

		self.assertEqual(D[0,0],0)
		self.assertEqual(D[0,1],1)
		self.assertEqual(D[0,2],2)
		self.assertEqual(D[0,3],2)
		self.assertEqual(D[1,2],1)
		self.assertEqual(D[1,3],1)
		self.assertEqual(D[2,3],float('infinity'))

		self.assertEqual(P[0,0],-1)
		self.assertEqual(P[0,1],0)
		self.assertEqual(P[0,2],1)
		self.assertEqual(P[0,3],1)
		self.assertEqual(P[1,2],1)
		self.assertEqual(P[1,3],1)
		self.assertEqual(P[2,3],-1)
		
	def test_disconnected(self):
		G = Graph()
		
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_node(4)

		D,P = all_pairs_bellman_ford_path_(G)
		
		self.assertEqual(D[0,3],float('infinity'))
		self.assertEqual(D[1,3],float('infinity'))
		self.assertEqual(D[2,3],float('infinity'))
		
		self.assertEqual(P[0,3],-1)
		self.assertEqual(P[1,3],-1)
		self.assertEqual(P[2,3],-1)

class AllPairsBellmanFordPathLengthTestCase(unittest.TestCase):
	
	def test_apdp_undirected(self):
		G = Graph()
		
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(2,4)

		D = all_pairs_bellman_ford_path_length(G)
		
		self.assertEqual(D[1][1],0)
		self.assertEqual(D[1][2],1)
		self.assertEqual(D[1][3],2)
		self.assertEqual(D[1][4],2)
		self.assertEqual(D[2][3],1)
		self.assertEqual(D[2][4],1)
		self.assertEqual(D[3][4],2)
		
	def test_apdp_directed(self):
		G = DiGraph()

		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(2,4)

		D = all_pairs_bellman_ford_path_length(G)

		self.assertEqual(D[1][1],0)
		self.assertEqual(D[1][2],1)
		self.assertEqual(D[1][3],2)
		self.assertEqual(D[1][4],2)
		self.assertEqual(D[2][3],1)
		self.assertEqual(D[2][4],1)
		self.assertEqual(D[3][4],float('infinity'))
		
	def test_disconnected(self):
		G = Graph()
		
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_node(4)

		D = all_pairs_bellman_ford_path_length(G)
		
		self.assertEqual(D[1][4],float('infinity'))
		self.assertEqual(D[2][4],float('infinity'))
		self.assertEqual(D[3][4],float('infinity'))

class AllPairsBellmanFordPathTestCase(unittest.TestCase):
	
	def test_apdp_undirected(self):
		G = Graph()
		
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(2,4)

		D = all_pairs_bellman_ford_path(G)
		
		self.assertEqual(D[1][1][0],0)
		self.assertEqual(D[1][2][0],1)
		self.assertEqual(D[1][3][0],2)
		self.assertEqual(D[1][4][0],2)
		self.assertEqual(D[2][3][0],1)
		self.assertEqual(D[2][4][0],1)
		self.assertEqual(D[3][4][0],2)
		
		self.assertEqual(D[1][1][1],None)
		self.assertEqual(D[1][2][1],1)
		self.assertEqual(D[1][3][1],2)
		self.assertEqual(D[1][4][1],2)
		self.assertEqual(D[2][3][1],2)
		self.assertEqual(D[2][4][1],2)
		self.assertEqual(D[3][4][1],2)
		
	def test_apdp_directed(self):
		G = DiGraph()

		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(2,4)

		D = all_pairs_bellman_ford_path(G)

		self.assertEqual(D[1][1][0],0)
		self.assertEqual(D[1][2][0],1)
		self.assertEqual(D[1][3][0],2)
		self.assertEqual(D[1][4][0],2)
		self.assertEqual(D[2][3][0],1)
		self.assertEqual(D[2][4][0],1)
		self.assertEqual(D[3][4][0],float('infinity'))

		self.assertEqual(D[1][1][1],None)
		self.assertEqual(D[1][2][1],1)
		self.assertEqual(D[1][3][1],2)
		self.assertEqual(D[1][4][1],2)
		self.assertEqual(D[2][3][1],2)
		self.assertEqual(D[2][4][1],2)
		self.assertEqual(D[3][4][1],None)
		
	def test_disconnected(self):
		G = Graph()
		
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_node(4)

		D = all_pairs_bellman_ford_path(G)
		
		self.assertEqual(D[1][4][0],float('infinity'))
		self.assertEqual(D[2][4][0],float('infinity'))
		self.assertEqual(D[3][4][0],float('infinity'))
		
		self.assertEqual(D[1][4][1],None)
		self.assertEqual(D[2][4][1],None)
		self.assertEqual(D[3][4][1],None)

class BellmanFordPathLengthTestCase(unittest.TestCase):
	
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

		D = bellman_ford_path_length(G, 'o')

		self.assertEqual(D['o'], 0)
		self.assertEqual(D['a'], 2)
		self.assertEqual(D['b'], 4)
		#self.assertEqual(D['d'], (8, 'e'))
		self.assertEqual(D['c'], 4)
		self.assertEqual(D['e'], 7)
		self.assertEqual(D['t'], 13)
		self.assertEqual(D['f'], 14)
        
		self.assertEqual(D['x'], float('infinity'))
		self.assertEqual(D['y'], float('infinity'))

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

		D = bellman_ford_path_length(G, 's')

		self.assertEqual(D['s'], 0)
		self.assertEqual(D['t'], 8)
		self.assertEqual(D['y'], 5)
		self.assertEqual(D['x'], 9)
		self.assertEqual(D['z'], 7)
        
		self.assertEqual(D['a'], float('infinity'))
		self.assertEqual(D['b'], float('infinity'))

class BellmanFordPathTestCase(unittest.TestCase):
	
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

		D = bellman_ford_path(G, 'o')

		self.assertEqual(D['o'], (0, None))
		self.assertEqual(D['a'], (2, 'o'))
		self.assertEqual(D['b'], (4, 'a'))
		#self.assertEqual(D['d'], (8, 'e'))
		self.assertEqual(D['c'], (4, 'o'))
		self.assertEqual(D['e'], (7, 'b'))
		self.assertEqual(D['t'], (13, 'd'))
		self.assertEqual(D['f'], (14, 'a'))
        
		self.assertEqual(D['x'], (float('infinity'),None)) 
		self.assertEqual(D['y'], (float('infinity'),None)) 

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

		D = bellman_ford_path(G, 's')

		self.assertEqual(D['s'], (0, None))
		self.assertEqual(D['t'], (8, 'y'))
		self.assertEqual(D['y'], (5, 's'))
		self.assertEqual(D['x'], (9, 't'))
		self.assertEqual(D['z'], (7, 'y'))
        
		self.assertEqual(D['a'], (float('infinity'),None)) 
		self.assertEqual(D['b'], (float('infinity'),None))

if __name__ == '__main__':
	unittest.main()
