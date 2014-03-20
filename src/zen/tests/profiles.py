from unittest import TestCase
import zen

class ProfileTestCase(TestCase):
	
	def test_all_sources(self):
		G = zen.DiGraph()
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(4,5)
		
		cp = zen.control.profile(G,normalized=False)
		self.assertEquals(cp,(2,0,0))
		
		cp = zen.control.profile(G)
		self.assertEquals(cp,(1,0,0))
		
	def test_an_external_dilation(self):
		G = zen.DiGraph()
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(2,4)

		cp = zen.control.profile(G,normalized=False)
		self.assertEquals(cp,(1,1,0))

		cp = zen.control.profile(G)
		self.assertEquals(cp,(0.5,0.5,0))
		
	def test_an_internal_dilation(self):
		G = zen.DiGraph()
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(3,4)
		G.add_edge(2,5)
		G.add_edge(5,4)

		cp = zen.control.profile(G,normalized=False)
		self.assertEquals(cp,(1,0,1))

		cp = zen.control.profile(G)
		self.assertEquals(cp,(0.5,0,0.5))
		
	def test_no_controls(self):
		G = zen.DiGraph()
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(3,4)
		G.add_edge(4,1)

		cp = zen.control.profile(G,normalized=False)
		self.assertEquals(cp,(0,0,0))

		cp = zen.control.profile(G)
		self.assertEquals(cp,(0,0,0))
	