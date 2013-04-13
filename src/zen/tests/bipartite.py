import unittest
import zen

class BipartiteTestCase(unittest.TestCase):

	def test_class_test(self):
		G = zen.BipartiteGraph()
		n1 = G.add_u_node(1)
		n2 = G.add_v_node(2)
	
		self.assertTrue(G.is_in_U(1))
		self.assertTrue(G.is_in_V(2))
		self.assertTrue(G.is_in_U_(n1))
		self.assertTrue(G.is_in_V_(n2))
		
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
		
	def test_uv_edges(self):
		G = zen.BipartiteGraph()
		G.add_u_node(1)
		G.add_u_node(3)
		G.add_u_node(5)
		G.add_v_node(2)
		G.add_v_node(4)
		
		G.add_edge(1,2,weight=1,data=True)
		G.add_edge(3,2,weight=1,data=True)
		G.add_edge(5,4,weight=1,data=True)
		
		# test each kind parameter combo
		for x,y in G.uv_edges():
			self.assertTrue(G.is_in_U(x))
			self.assertTrue(G.is_in_V(y))

		for x,y,data in G.uv_edges(data=True):
			self.assertTrue(G.is_in_U(x))
			self.assertTrue(G.is_in_V(y))
			self.assertEquals(data,True)
		
		for x,y,data,weight in G.uv_edges(weight=True,data=True):
			self.assertTrue(G.is_in_U(x))
			self.assertTrue(G.is_in_V(y))
			self.assertEquals(data,True)
			self.assertEquals(weight,1)
			
		return
	
	def test_uv_edges_iter(self):
		G = zen.BipartiteGraph()
		G.add_u_node(1)
		G.add_u_node(3)
		G.add_u_node(5)
		G.add_v_node(2)
		G.add_v_node(4)

		G.add_edge(1,2,weight=1,data=True)
		G.add_edge(3,2,weight=1,data=True)
		G.add_edge(5,4,weight=1,data=True)

		# test each kind parameter combo
		for x,y in G.uv_edges_iter():
			self.assertTrue(G.is_in_U(x))
			self.assertTrue(G.is_in_V(y))

		for x,y,data in G.uv_edges_iter(data=True):
			self.assertTrue(G.is_in_U(x))
			self.assertTrue(G.is_in_V(y))
			self.assertEquals(data,True)

		for x,y,data,weight in G.uv_edges_iter(weight=True,data=True):
			self.assertTrue(G.is_in_U(x))
			self.assertTrue(G.is_in_V(y))
			self.assertEquals(data,True)
			self.assertEquals(weight,1)

		return			
		
if __name__ == '__main__':
	unittest.main()