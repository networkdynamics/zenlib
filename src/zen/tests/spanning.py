from zen import *
import unittest

class MinimumSpanningTreeTestCase(unittest.TestCase):
	
	def test_simple(self):
		"""
			Test taken from http://en.wikipedia.org/wiki/Prim%27s_algorithm
		"""
		G = Graph()
		G.add_node('A', data='AAA')
		G.add_node('B', data='BBB')
		G.add_edge_(G.node_idx('A'), G.node_idx('B'), data='AB', weight=7)
		G.add_edge('A', 'D', 'AD', 5)
		G.add_edge('B', 'C', 'BC', 8)
		G.add_edge('D', 'B', 'DB', 9)
		G.add_edge('D', 'E', 'DE', 15)
		G.add_edge('D', 'F', 'DF', 6)
		G.add_edge('F', 'E', 'FE', 8)
		G.add_edge('F', 'G', 'FG', 11)
		G.add_edge('G', 'E', 'GE', 9)
		G.add_edge('B', 'E', 'BE', 7)
		G.add_edge('E', 'C', 'EC', 5)
		
		Gnew = spanning.minimum_spanning_tree(G)
		
		# We only have 6 edges in the MST
		self.assertTrue(Gnew.num_edges == 6)
		
		# Node data transfered in the new graph
		self.assertTrue(Gnew.node_data_(Gnew.node_idx('A')) == 'AAA')
		self.assertTrue(Gnew.node_data_(Gnew.node_idx('B')) == 'BBB')
		
		# We have these edges
		self.assertTrue(Gnew.has_edge('A', 'B'))
		self.assertTrue(Gnew.has_edge('A', 'D'))
		self.assertTrue(Gnew.has_edge('D', 'F'))
		self.assertTrue(Gnew.has_edge('E', 'G'))
		self.assertTrue(Gnew.has_edge('B', 'E'))
		self.assertTrue(Gnew.has_edge('E', 'C'))
		
		# The edge data was conserved.
		self.assertTrue(Gnew.edge_data('A', 'B') == 'AB')
		self.assertTrue(Gnew.edge_data('A', 'D') == 'AD')
		self.assertTrue(Gnew.edge_data('D', 'F') == 'DF')
		self.assertTrue(Gnew.edge_data('G', 'E') == 'GE')
		self.assertTrue(Gnew.edge_data('B', 'E') == 'BE')
		self.assertTrue(Gnew.edge_data('E', 'C') == 'EC')
		
		# Edge weights were conserved
		self.assertTrue(Gnew.weight('A', 'B') == 7)
		self.assertTrue(Gnew.weight('A', 'D') == 5)
		self.assertTrue(Gnew.weight('D', 'F') == 6)
		self.assertTrue(Gnew.weight('E', 'G') == 9)
		self.assertTrue(Gnew.weight('B', 'E') == 7)
		self.assertTrue(Gnew.weight('E', 'C') == 5)
	
	def test_simple2(self):
		"""
		Taken from http://www.cs.uiuc.edu/~jeffe/teaching/algorithms/notes/12-mst.pdf
		"""
		G = Graph()
		
		G.add_edge(1, 3, None, 5)
		G.add_edge(1, 2, None, 8)
		G.add_edge(2, 3, None, 10)
		G.add_edge(2, 4, None, 2)
		G.add_edge(3, 4, None, 3)
		G.add_edge(4, 5, None, 12)
		G.add_edge(5, 2, None, 18)
		G.add_edge(4, 6, None, 30)
		G.add_edge(3, 6, None, 16)
		G.add_edge(5, 7, None, 4)
		G.add_edge(4, 7, None, 14)
		G.add_edge(7, 6, None, 26)
		
		Gnew = spanning.minimum_spanning_tree(G)
		
		self.assertTrue(Gnew.num_edges == 6)
		self.assertTrue(Gnew.has_edge(1,3))
		self.assertTrue(Gnew.has_edge(2,4))
		self.assertTrue(Gnew.has_edge(4,3))
		self.assertTrue(Gnew.has_edge(3,6))
		self.assertTrue(Gnew.has_edge(4,5))
		self.assertTrue(Gnew.has_edge(5,7))
		
		# Edge weights were conserved
		self.assertTrue(Gnew.weight(1, 3) == 5)
		self.assertTrue(Gnew.weight(2, 4) == 2)
		self.assertTrue(Gnew.weight(4, 3) == 3)
		self.assertTrue(Gnew.weight(3, 6) == 16)
		self.assertTrue(Gnew.weight(4, 5) == 12)
		self.assertTrue(Gnew.weight(5, 7) == 4)
		
	def test_empty(self):
		G = Graph()
		Gnew = spanning.minimum_spanning_tree(G)
		self.assertTrue(Gnew.num_nodes == 0)
				
if __name__ == '__main__':
	unittest.main()
