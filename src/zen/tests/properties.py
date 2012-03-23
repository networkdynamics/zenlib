import unittest
import random
import types
import os.path

from zen import *

class PropertiesTestCase(unittest.TestCase):
	
	def test_ug_components(self):
		G = Graph()
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(4,5)
		G.add_edge(5,6)
		G.add_node(7)
		
		cp = components(G)
		self.assertEqual(len(cp),3)
	
	# def test_dg_components(self):
	# 	G = DiGraph()
	# 	G.add_edge(1,2)
	# 	G.add_edge(2,3)
	# 	G.add_edge(4,5)
	# 	G.add_edge(5,6)
	# 	G.add_node(7)
	# 
	# 	cp = components(G)
	# 	self.assertEqual(len(cp),3)
	
	def test_diameter(self):
		G = DiGraph()
		n1 = G.add_edge('x','y')
		n2 = G.add_edge('y','z')
		n3 = G.add_edge('z','a')
		G.add_edge('z','y')

		self.assertEqual(diameter(G),3)

	def test_ddist_u1(self):
		G = Graph()
		G.add_edge('x','y')
		G.add_edge('x','z')
		G.add_edge('x','a')
		G.add_edge('a','z')
		
		R = ddist(G,normalize=False)
		self.assertEqual(len(R),4)
		self.assertEqual(R[3],1)
		
		R = ddist(G,normalize=True)
		self.assertEqual(len(R),4)
		self.assertEqual(R[3],0.25)
		
		try:
			R = ddist(G,direction=IN_DIR,normalize=False)
			self.fail('direction cannot be specified on an undirected graph')
		except ZenException:
			pass
	
	def test_ddist_d1(self):
		G = DiGraph()
		G.add_edge('x','y')
		G.add_edge('x','z')
		G.add_edge('x','a')
		G.add_edge('a','z')
		
		R = ddist(G,normalize=False)
		self.assertEqual(len(R),4)
		self.assertEqual(R[3],1)
		
		R = ddist(G,normalize=True)
		self.assertEqual(len(R),4)
		self.assertEqual(R[3],0.25)
		
		R = ddist(G,direction=IN_DIR,normalize=False)
		self.assertEqual(len(R),3)
		self.assertEqual(R[0],1)
		self.assertEqual(R[1],2)
		
	def test_cum_ddist_d1(self):
		G = DiGraph()
		G.add_edge('x','y')
		G.add_edge('x','z')
		G.add_edge('x','a')
		G.add_edge('a','z')

		R = cddist(G)
		self.assertEqual(len(R),4)
		self.assertEqual(R[0],0)
		self.assertEqual(R[3],1)
		self.assertEqual(R[1],0.25)

		# R = ddist(G,direction=IN_DIR,normalize=False)
		# 		self.assertEqual(len(R),3)
		# 		self.assertEqual(R[0],1)
		# 		self.assertEqual(R[1],2)
		# 		
	# def test_diameter_gtm1(self):
	# 	"""
	# 	This is more of a performance test.  We don't include it in the usual test.
	# 	"""
	# 	G = read_edgelist(os.path.join(os.path.dirname(__file__),'gtm1.elist'),True)
	# 	
	# 	self.assertEqual(diameter(G), 13)