import zen
import unittest

class NumMinControlsTestCase(unittest.TestCase):
	
	def test_1(self):
		G = zen.DiGraph()
		G.add_edge(1,2)
		G.add_edge(3,4)
		
		n = zen.control.num_min_controls(G)
		
		self.assertEqual(n,2)
		
	def test_2(self):
		G = zen.DiGraph()
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(2,4)
		
		n = zen.control.num_min_controls(G)
		
		self.assertEqual(n,2)
		
	def test_cycle_1(self):
		G = zen.DiGraph()
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(3,1)
		
		n = zen.control.num_min_controls(G)
		
		self.assertEqual(n,1)