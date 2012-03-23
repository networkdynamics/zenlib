import unittest
from zen import *

class ForceatlasTestCase(unittest.TestCase):
	def test_simple(self):
		g = Graph()
		g.add_edge(1,2)
		g.add_edge(2,3)
		g.add_edge(3,4)
		g.add_edge(4,1)
		view = layout.forceatlas(g)
	def test_empty_graph(self):
		g = Graph()
		view = layout.forceatlas(g)
	def test_self_edges(self):
		g = Graph()
		g.add_edge(1,2)
		g.add_edge(2,3)
		g.add_edge(3,4)
		g.add_edge(4,4)
		g.add_edge(4,1)
		view = layout.forceatlas(g)
	def test_no_edges(self):
		g = Graph()
		g.add_node()
		g.add_node()
		view = layout.forceatlas(g)