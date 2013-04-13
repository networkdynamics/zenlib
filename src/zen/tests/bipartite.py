import unittest
import zen

class BipartiteTestCase(unittest.TestCase):
	
	def test_copy(self):
		G = zen.BipartiteGraph()
		G.add_u_node(1)
		G.add_v_node(2)
		
		G2 = G.copy()
		
		self.assertIsInstance(G2,zen.BipartiteGraph)

	def test_uv_endpoints(self):
		G = zen.BipartiteGraph()
		n2 = G.add_v_node(2)
		n1 = G.add_u_node(1)
		eidx = G.add_edge(1,2)
		
		self.assertEquals((1,2),G.uv_endpoints(eidx))
		self.assertEquals((n1,n2),G.uv_endpoints_(eidx))

if __name__ == '__main__':
	unittest.main()