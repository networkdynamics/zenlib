import unittest
import random
import types
import os.path

from zen import *

class ComponentsTestCase(unittest.TestCase):
	
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