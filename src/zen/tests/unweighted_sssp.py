import unittest
from zen import *

class AllPairsUWShortestPathLength_TestCase(unittest.TestCase):
	
	def test_apdp_undirected(self):
		G = Graph()
		
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(2,4)

		D = all_pairs_shortest_path_length_(G)
		
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

		D = all_pairs_shortest_path_length_(G)

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

		D = all_pairs_shortest_path_length_(G)
		
		self.assertEqual(D[0,3],float('infinity'))
		self.assertEqual(D[1,3],float('infinity'))
		self.assertEqual(D[2,3],float('infinity'))

class AllPairsUWShortestPath_TestCase(unittest.TestCase):
	
	def test_apdp_undirected(self):
		G = Graph()
		
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(2,4)

		D,P = all_pairs_shortest_path_(G)
		
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

		D,P = all_pairs_shortest_path_(G)

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

		D,P = all_pairs_shortest_path_(G)
		
		self.assertEqual(D[0,3],float('infinity'))
		self.assertEqual(D[1,3],float('infinity'))
		self.assertEqual(D[2,3],float('infinity'))
		
		self.assertEqual(P[0,3],-1)
		self.assertEqual(P[1,3],-1)
		self.assertEqual(P[2,3],-1)

class AllPairsUWShortestPathLengthTestCase(unittest.TestCase):
	
	def test_apdp_undirected(self):
		G = Graph()
		
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(2,4)

		D = all_pairs_shortest_path_length(G)
		
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

		D = all_pairs_shortest_path_length(G)

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

		D = all_pairs_shortest_path_length(G)
		
		self.assertEqual(D[1][4],float('infinity'))
		self.assertEqual(D[2][4],float('infinity'))
		self.assertEqual(D[3][4],float('infinity'))

class AllPairsUWShortestPathTestCase(unittest.TestCase):
	
	def test_apdp_undirected(self):
		G = Graph()
		
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(2,4)

		D = all_pairs_shortest_path(G)
		
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

		D = all_pairs_shortest_path(G)

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

		D = all_pairs_shortest_path(G)
		
		self.assertEqual(D[1][4][0],float('infinity'))
		self.assertEqual(D[2][4][0],float('infinity'))
		self.assertEqual(D[3][4][0],float('infinity'))
		
		self.assertEqual(D[1][4][1],None)
		self.assertEqual(D[2][4][1],None)
		self.assertEqual(D[3][4][1],None)

class UWSSSPLengthTestCase(unittest.TestCase):
	
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

		D = single_source_shortest_path_length(G, 'o')

		self.assertEqual(D['o'], 0)
		self.assertEqual(D['a'], 1)
		self.assertEqual(D['b'], 1)
		self.assertEqual(D['d'], 2)
        
		self.assertEqual(D['x'], float('infinity')) 
		self.assertEqual(D['y'], float('infinity')) 

	def test_unreachable(self):
        
		G = DiGraph()
		G.add_edge('s', 't', None, 10)
		G.add_edge('x', 'z', None, 2)

		d = single_source_shortest_path_length(G, 's', 'x')
		
		self.assertEquals(float('infinity'), d)

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

		D = single_source_shortest_path_length(G, 's')

		self.assertEqual(D['s'], 0)
		self.assertEqual(D['y'], 1)
        
		self.assertEqual(D['a'], float('infinity')) 
		self.assertEqual(D['b'], float('infinity')) 

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

		d = single_source_shortest_path_length(G, 's', 't') # x should not be found
       	
		self.assertEquals(2, d)

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
        
		self.assertEqual(D['x'], (float('infinity'),None)) 
		self.assertEqual(D['y'], (float('infinity'),None)) 

	def test_unreachable(self):
        
		G = DiGraph()
		G.add_edge('s', 't', None, 10)
		G.add_edge('x', 'z', None, 2)

		d,p = single_source_shortest_path(G, 's', 'x')
		
		self.assertEquals(float('infinity'), d)
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
        
		self.assertEqual(D['a'], (float('infinity'),None)) 
		self.assertEqual(D['b'], (float('infinity'),None)) 

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
