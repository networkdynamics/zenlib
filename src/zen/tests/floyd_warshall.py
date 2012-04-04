import unittest
from zen import *

class FloydWarshallPath_TestCase(unittest.TestCase):
	
	def test_simple_directed(self):
		
		G = DiGraph()
		G.add_edge('x','y')
		G.add_edge('y','z')
		G.add_edge('z','x')
		
		D,P = floyd_warshall_path_(G)
		
		self.assertEqual(D[0,1],1)
		self.assertEqual(D[0,2],2)
		self.assertEqual(D[2,0],1)
		self.assertEqual(D[0,0],0)
		
		self.assertEqual(P[0,1],0)
		self.assertEqual(P[0,2],1)
		self.assertEqual(P[2,0],2)
		self.assertEqual(P[0,0],-1)
		
	def test_simple_undirected(self):

		G = Graph()
		G.add_edge('x','y')
		G.add_edge('y','z')
		G.add_edge('z','x')

		D,P = floyd_warshall_path_(G)

		self.assertEqual(D[0,1],1)
		self.assertEqual(D[0,2],1)
		self.assertEqual(D[2,0],1)
		self.assertEqual(D[0,0],0)
		
		self.assertEqual(P[0,1],0)
		self.assertEqual(P[0,2],0)
		self.assertEqual(P[2,0],2)
		self.assertEqual(P[0,0],-1)
		
	def test_disconnected(self):
		
		G = Graph()
		G.add_edge(1,2)
		G.add_node(3)
		
		D,P = floyd_warshall_path_(G)
		
		self.assertEqual(D[0,1],1)
		self.assertEqual(D[0,2],float('infinity'))
		self.assertEqual(D[1,2],float('infinity'))
		
		self.assertEqual(P[0,1],0)
		self.assertEqual(P[0,2],-1)
		self.assertEqual(P[1,2],-1)

class FloydWarshallPathLength_TestCase(unittest.TestCase):
	
	def test_simple_directed(self):
		
		G = DiGraph()
		G.add_edge('x','y')
		G.add_edge('y','z')
		G.add_edge('z','x')
		
		D = floyd_warshall_path_length_(G)
		
		self.assertEqual(D[0,1],1)
		self.assertEqual(D[0,2],2)
		self.assertEqual(D[2,0],1)
		self.assertEqual(D[0,0],0)
		
	def test_simple_undirected(self):

		G = Graph()
		G.add_edge('x','y')
		G.add_edge('y','z')
		G.add_edge('z','x')

		D = floyd_warshall_path_length_(G)

		self.assertEqual(D[0,1],1)
		self.assertEqual(D[0,2],1)
		self.assertEqual(D[2,0],1)
		self.assertEqual(D[0,0],0)
		
	def test_disconnected(self):
		
		G = Graph()
		G.add_edge(1,2)
		G.add_node(3)
		
		D = floyd_warshall_path_length_(G)
		
		self.assertEqual(D[0,1],1)
		self.assertEqual(D[0,2],float('infinity'))
		self.assertEqual(D[1,2],float('infinity'))

class FloydWarshallPathTestCase(unittest.TestCase):
	
	def test_simple_directed(self):
		
		G = DiGraph()
		G.add_edge('x','y')
		G.add_edge('y','z')
		G.add_edge('z','x')
		
		D = floyd_warshall_path(G)
		
		self.assertEqual(D['x']['y'][0],1)
		self.assertEqual(D['x']['z'][0],2)
		self.assertEqual(D['z']['x'][0],1)
		self.assertEqual(D['x']['x'][0],0)
		
		self.assertEqual(D['x']['y'][1],'x')
		self.assertEqual(D['x']['z'][1],'y')
		self.assertEqual(D['z']['x'][1],'z')
		self.assertEqual(D['x']['x'][1],None)
		
	def test_simple_undirected(self):

		G = Graph()
		G.add_edge('x','y')
		G.add_edge('y','z')
		G.add_edge('z','x')

		D = floyd_warshall_path(G)

		self.assertEqual(D['x']['y'][0],1)
		self.assertEqual(D['x']['z'][0],1)
		self.assertEqual(D['z']['x'][0],1)
		self.assertEqual(D['x']['x'][0],0)
		
		self.assertEqual(D['x']['y'][1],'x')
		self.assertEqual(D['x']['z'][1],'x')
		self.assertEqual(D['z']['x'][1],'z')
		self.assertEqual(D['x']['x'][1],None)
		
	def test_disconnected(self):
		
		G = Graph()
		G.add_edge(1,2)
		G.add_node(3)
		
		D = floyd_warshall_path(G)
		
		self.assertEqual(D[1][2][0],1)
		self.assertEqual(D[1][3][0],float('infinity'))
		self.assertEqual(D[2][3][0],float('infinity'))
		
		self.assertEqual(D[1][2][1],1)
		self.assertEqual(D[1][3][1],None)
		self.assertEqual(D[2][3][1],None)

class FloydWarshallPathLengthTestCase(unittest.TestCase):
	
	def test_simple_directed(self):
		
		G = DiGraph()
		G.add_edge('x','y')
		G.add_edge('y','z')
		G.add_edge('z','x')
		
		D = floyd_warshall_path_length(G)
		
		self.assertEqual(D['x']['y'],1)
		self.assertEqual(D['x']['z'],2)
		self.assertEqual(D['z']['x'],1)
		self.assertEqual(D['x']['x'],0)
		
	def test_simple_undirected(self):

		G = Graph()
		G.add_edge('x','y')
		G.add_edge('y','z')
		G.add_edge('z','x')

		D = floyd_warshall_path_length(G)

		self.assertEqual(D['x']['y'],1)
		self.assertEqual(D['x']['z'],1)
		self.assertEqual(D['z']['x'],1)
		self.assertEqual(D['x']['x'],0)
		
	def test_disconnected(self):
		
		G = Graph()
		G.add_edge(1,2)
		G.add_node(3)
		
		D = floyd_warshall_path_length(G)
		
		self.assertEqual(D[1][2],1)
		self.assertEqual(D[1][3],float('infinity'))
		self.assertEqual(D[2][3],float('infinity'))