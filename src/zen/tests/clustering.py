import unittest
import random
from zen import *

# class WeakClusteringCoefficientTestCase(unittest.TestCase):
# 	
# 	def test_simple(self):
# 		
# 		G = DiGraph()
# 		G.add_edge('x','y')
# 		G.add_edge('x','z')
# 		G.add_edge('y','z')
# 		
# 		x = clustering.weak(G,nbunch=['x'],avg=True)
# 		self.assertEqual(x,0.5)
# 		
# 		G.add_edge('z','y')
# 		x = clustering.weak(G,nbunch=['x'],avg=True)
# 		self.assertEqual(x,1.0)
# 		
# 		x = clustering.weak(G,avg=True)
# 		self.assertAlmostEqual(x,0.333333333333333)
# 	
# 	def test_bunch(self):
# 		G = DiGraph()
# 		nodes = range(30)
# 		for i in range(1000):
# 			n1 = random.choice(nodes)
# 			n2 = random.choice(nodes)
# 			
# 			if n1 != n2:
# 				if (n1 not in G or n2 not in G) or not G.has_edge(n1,n2):
# 					G.add_edge(n1,n2)
# 			
# 		wccs = clustering.weak(G,nbunch=nodes)
# 		
# 		for n,w in zip(nodes,wccs):
# 			self.assertEqual(clustering.weak(G,nbunch=[n])[0],w)
# 
# 		
# 	def test_zero_outdegree(self):
# 		
# 		G = DiGraph()
# 		G.add_edge('x','y')
# 		
# 		x = clustering.weak(G,nbunch=['y'],avg=True)
# 		self.assertEqual(x,0)
# 		
# 	def test_simple2(self):
# 		G = DiGraph()
# 		n1 = G.add_node()
# 		n2 = G.add_node()
# 		G.add_edge_(n1,n2)
# 		
# 		for i in range(2,100):
# 			n = G.add_node()
# 			G.add_edge_(n1,n)
# 			G.add_edge_(n,n-1)
# 		
# 		G.add_edge_(n2,99)	
# 		x = clustering.weak(G,nbunch_=[n1])
		
class LocalClusteringCoefficientTestCase(unittest.TestCase):

	def test_simple(self):

		G = DiGraph()
		G.add_edge('x','y')
		G.add_edge('x','z')
		G.add_edge('y','z')

		x = clustering.lcc(G) #,nbunch=['x'],avg=True)
		self.assertEqual(x['x'],0.5)

		G.add_edge('z','y')
		x = clustering.lcc(G) #,nbunch=['x'],avg=True)
		self.assertEqual(x['x'],1.0)

		x = clustering.ncc(G) #,avg=True)
		self.assertAlmostEqual(x,0.333333333333333)

	def test_zero_outdegree(self):

		G = DiGraph()
		G.add_edge('x','y')

		x = clustering.lcc(G) #,nbunch=['y'],avg=True)
		self.assertEqual(x['y'],0)

	def test_simple2(self):
		G = DiGraph()
		n1 = G.add_node()
		n2 = G.add_node()
		G.add_edge_(n1,n2)

		for i in range(2,100):
			n = G.add_node()
			G.add_edge_(n1,n)
			G.add_edge_(n,n-1)

		G.add_edge_(n2,99)	
		x = clustering.lcc(G)
		
class GlobalClusteringCoefficientTestCase(unittest.TestCase):

	def test_simple_dir(self):

		G = DiGraph()
		G.add_edge('x','y')
		G.add_edge('x','z')
		G.add_edge('y','z')

		x = clustering.gcc(G)
		self.assertEqual(x,0.5)

		G.add_edge('z','y')
		x = clustering.gcc(G)
		self.assertEqual(x,1.0)

	def test_simple_udir(self):

		G = Graph()
		G.add_edge('x','y')
		G.add_edge('x','z')
		G.add_edge('y','z')

		x = clustering.gcc(G)
		self.assertEqual(x,1.0)

		G.add_edge('z','w')
		x = clustering.gcc(G)
		self.assertAlmostEqual(x,3.0/5.0)

	def test_zero_outdegree(self):

		G = DiGraph()
		G.add_edge('x','y')

		x = clustering.gcc(G)
		self.assertEqual(x,0)

	def test_simple2(self):
		G = DiGraph()
		n1 = G.add_node()
		n2 = G.add_node()
		G.add_edge_(n1,n2)

		for i in range(2,100):
			n = G.add_node()
			G.add_edge_(n1,n)
			G.add_edge_(n,n-1)

		G.add_edge_(n2,99)	
		x = clustering.gcc(G)
		
# class TTClusteringCoefficientTestCase(unittest.TestCase):
# 
# 	def test_simple(self):
# 
# 		G = DiGraph()
# 		G.add_edge('x','y')
# 		G.add_edge('x','z')
# 		G.add_edge('y','z')
# 
# 		x = clustering.tt(G)
# 		self.assertEqual(x,1.0)
# 
# 		G.add_edge('z','y')
# 		x = clustering.tt(G)
# 		self.assertEqual(x,1.0)
# 
# 	def test_zero_outdegree(self):
# 
# 		G = DiGraph()
# 		G.add_edge('x','y')
# 
# 		x = clustering.tt(G)
# 		self.assertEqual(x,0)
# 
# 	def test_simple2(self):
# 		G = DiGraph()
# 		n1 = G.add_node()
# 		n2 = G.add_node()
# 		G.add_edge_(n1,n2)
# 
# 		for i in range(2,100):
# 			n = G.add_node()
# 			G.add_edge_(n1,n)
# 			G.add_edge_(n,n-1)
# 
# 		G.add_edge_(n2,99)	
# 		x = clustering.tt(G)