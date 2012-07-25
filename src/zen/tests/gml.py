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
		
	def test_weight_fxn(self):
		fname = path.join(path.dirname(__file__),'test3.gml')
		G = gml.read(fname,weight_fxn=lambda data:data['value'])
		
		self.assertEqual(len(G),3)
		self.assertEqual(G.size(),2)
		
		self.assertEqual(G.weight('N1','N2'),2)
		self.assertEqual(G.weight('N2','N3'),3)

class GMLWriteTestCase(unittest.TestCase):
	def test_empty_graph(self):
		test_path = path.dirname(__file__) + '/test4.gml'
		G = Graph()
		gml.write(G, test_path)
		G_ = gml.read(test_path)
		self.assertEqual(type(G),Graph)
		self.assertEqual(len(G),0)
		self.assertEqual(G.size(),0)
		# E: Delete this file once the test is done.
		
	def test_empty_digraph(self):	
		test_path = path.dirname(__file__) + '/test4.gml'
		G = DiGraph()
		gml.write(G, test_path)
		G_ = gml.read(test_path)
		self.assertEqual(type(G),DiGraph)
		self.assertEqual(len(G),0)
		self.assertEqual(G.size(),0)
		# E: Delete this file once the test is done.
		
	def test_empty_BipartiteGraph(self):
		test_path = path.dirname(__file__) + '/test4.gml'
		G = BipartiteGraph()
		gml.write(G, test_path)
		G_copy = gml.read(test_path)
		# TODO: consider recognizing a bipartite parameter during read
		# self.assertEqual(type(G_copy),BipartiteGraph)
		self.assertEqual(len(G_copy),0)
		self.assertEqual(G_copy.size(),0)
		# E: Delete this file once the test is done.
		
	def test_graph_obj_data(self):
		test_path = path.dirname(__file__) + '/test5'
		# commented nobj, and datum below would be good in test, but are not yet supported.
		# nobj = 'she said "&!*<{[(\\"'
		# nobj = (1, 'A', ('nested', 'tuple'))
		nobj = "aString"
		# datum = {'alist':[1,'two', 3.14], "nested_dict":{'special_chars':'she said "&!*<{[(\'"}}
		ndatum = "bString"
		edatum = "cString"
		G = Graph()
		G.add_node(nobj=nobj, data=ndatum)
		G.add_node('A')
		G.add_node('B')
		G.add_edge(nobj, 'A', data=edatum)
		G.add_edge('A', 'B')
		gml.write(G, test_path)
		G_copy = gml.read(test_path)
		self.assertEqual(type(G_copy), Graph)
		self.assertEqual(len(G_copy), 3)
		self.assertEqual(G_copy.size(), 2)
		self.assertTrue(G_copy.has_edge(nobj, 'A'))				
		self.assertTrue(G_copy.has_edge('A', 'B'))				
		self.assertEqual(G_copy.node_data(nobj)['zen_data'], ndatum)			
		self.assertEqual(G_copy.edge_data(nobj, 'A')['zen_data'], edatum)
		# E: Delete this file once the test is done.
		
	def test_graph_weight(self):
		test_path = path.dirname(__file__) + '/test6.gml'
		G = Graph()
		G.add_node('A')
		G.add_node('B')
		G.add_node('C')
		G.add_edge('A', 'B', weight=0)
		G.add_edge('B', 'C', weight=10)
		G.add_edge('C', 'A', weight=3.141592)
		gml.write(G, test_path)
		G_copy = gml.read(test_path,weight_fxn=lambda data:data['weight'])
		self.assertEqual(G_copy.weight('A','B'),0)
		self.assertEqual(G_copy.weight('B','C'),10)		
		self.assertEqual(G_copy.weight('C','A'),3.141592)
		# E: Delete this file once the test is done.
			
		
	def test_large_graph(self):
		test_path = path.dirname(__file__) + '/test6.gml'
		G = Graph()
		G.add_nodes(100000)
		for i in range (99999): # make a large line
			G.add_edge_(i, i+1)
		gml.write(G, test_path)
		G_copy = gml.read(test_path)
		self.assertEqual(len(G), 100000)
		preserved_edges = True
		for i in range(99999):
			# E: use 'and' not '&'
			preserved_edges = preserved_edges & G.has_edge_(i, i+1)
		self.assertTrue(preserved_edges)
		self.assertEqual(G.size(), 99999)
		# E: Delete this file once the test is done.
		

if __name__ == '__main__':
	unittest.main()