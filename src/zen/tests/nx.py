import unittest
import random
import types
import networkx

from zen import *

class ToNetworkXTestCase(unittest.TestCase):
	
	def test_return_types(self):
		Gu = Graph()
		Gd = DiGraph()
		
		self.assertTrue(type(to_networkx(Gu)) == networkx.Graph)
		self.assertTrue(type(to_networkx(Gd)) == networkx.DiGraph)
	
	def test_basic_udir(self):
		G = Graph()
		G.add_edge(1,'hello')
		G.add_edge('hello',0)
		
		Gnx = to_networkx(G)
		
		assert 1 in Gnx
		assert 'hello' in Gnx
		assert True in Gnx
		
		assert Gnx.has_edge(1,'hello')
		assert not Gnx.has_edge(1,0)
		assert Gnx.has_edge('hello',0)
		assert Gnx.has_edge(0,'hello')
		
	def test_basic_dir(self):
		G = DiGraph()
		G.add_edge(1,'hello')
		G.add_edge('hello',0)
		
		Gnx = to_networkx(G)
		
		assert 1 in Gnx
		assert 'hello' in Gnx
		assert True in Gnx
		
		assert Gnx.has_edge(1,'hello')
		assert not Gnx.has_edge(1,0)
		assert Gnx.has_edge('hello',0)
		assert not Gnx.has_edge(0,'hello')
		
if __name__ == '__main__':
	unittest.main()
