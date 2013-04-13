import unittest
import zen

class BipartiteTestCase(unittest.TestCase):
	
	def test_copy(self):
		G = zen.BipartiteGraph()
		G.add_u_node(1)
		G.add_v_node(2)
		
		G2 = G.copy()
		
		self.assertIsInstance(G2,zen.BipartiteGraph)

if __name__ == '__main__':
	unittest.main()