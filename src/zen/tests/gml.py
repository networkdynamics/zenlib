from zen import *
import unittest
import os
import os.path as path
import tempfile

class GMLReadTestCase(unittest.TestCase):
		
	def test_read_directed_test1(self):
		fname = path.join(path.dirname(__file__),'test1.gml')
		G = gml.read(fname)
		
		self.assertEqual(len(G),3)
		self.assertEqual(G.size(),2)
		
		self.assertEqual(type(G),DiGraph)
		self.assertTrue(G.has_edge('N1','N2'))
		self.assertTrue(G.has_edge('N2','N3'))
		self.assertFalse(G.has_edge('N1','N3'))
		self.assertFalse(G.has_edge('N3','N2'))
		
		self.assertEqual(G.node_idx('N1'),1)
		self.assertEqual(G.node_idx('N2'),2)
		self.assertEqual(G.node_idx('N3'),3)
		
		self.assertEqual(G.node_data('N1')['sample1'],42)
		self.assertEqual(G.node_data('N2')['sample2'],42.1)
		self.assertEqual(G.node_data('N3')['sample3'],'HELLO WORLD')
		
		self.assertEqual(G.edge_data('N1','N2')['label'],'Edge from node 1 to node 2')
		
	def test_read_undirected_test1(self):
		fname = path.join(path.dirname(__file__),'test2.gml')
		G = gml.read(fname)
		
		self.assertEqual(len(G),3)
		self.assertEqual(G.size(),2)
		
		self.assertEqual(type(G),Graph)
		self.assertTrue(G.has_edge('N1','N2'))
		self.assertTrue(G.has_edge('N2','N3'))
		self.assertFalse(G.has_edge('N1','N3'))
		self.assertTrue(G.has_edge('N3','N2'))
		
		self.assertEqual(G.node_idx('N1'),1)
		self.assertEqual(G.node_idx('N2'),2)
		self.assertEqual(G.node_idx('N3'),3)
		
		self.assertEqual(G.node_data('N1')['sample1'],42)
		self.assertEqual(G.node_data('N2')['sample2'],42.1)
		self.assertEqual(G.node_data('N3')['sample3'],'HELLO WORLD')
		
		self.assertEqual(G.edge_data('N1','N2')['label'],'Edge from node 1 to node 2')
		
	def test_list_variables(self):
		fname = path.join(path.dirname(__file__),'test3.gml')
		G = gml.read(fname)
		
		self.assertEqual(len(G),3)
		self.assertEqual(G.size(),2)
		
		self.assertEqual(G.node_data('N1')['listVar'],[1,'a',3.2])
		
if __name__ == '__main__':
	unittest.main()