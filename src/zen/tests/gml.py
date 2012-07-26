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
		test_file = path.dirname(__file__) + '/test4.gml'
		G = Graph()
		gml.write(G, test_file)
		G_ = gml.read(test_file)
		self.assertEqual(type(G),Graph)
		self.assertEqual(len(G),0)
		self.assertEqual(G.size(),0)
		os.remove(test_file)
		
	def test_empty_digraph(self):	
		test_file = path.dirname(__file__) + '/test4.gml'
		G = DiGraph()
		gml.write(G, test_file)
		G_ = gml.read(test_file)
		self.assertEqual(type(G),DiGraph)
		self.assertEqual(len(G),0)
		self.assertEqual(G.size(),0)
		os.remove(test_file)
		
	def test_empty_BipartiteGraph(self):
		test_file = path.dirname(__file__) + '/test4.gml'
		G = BipartiteGraph()
		gml.write(G, test_file)
		G_copy = gml.read(test_file)
		# TODO: consider recognizing a bipartite parameter during read
		# self.assertEqual(type(G_copy),BipartiteGraph)
		self.assertEqual(len(G_copy),0)
		self.assertEqual(G_copy.size(),0)
		os.remove(test_file)
		
	def test_graph_obj_data(self):
		test_file = path.dirname(__file__) + '/test5.gml'
		
		# nobj = 'she said "&!*<{[(\\"'
		# nobj = (1, 'A', ('nested', 'tuple'))
		# datum = {'alist':[1,'two', 3.14], "nested_dict":{'special_chars':'she said "&!*<{[(\'"}}
		
		G = Graph()
		G.add_node(nobj='A', data='B')
		G.add_node(nobj=1, data=2)				# causes error: zen/io/gml.py ln# 268 enforces this must be str
		G.add_node(nobj=True, data=False)
		G.add_node(nobj=2**33, data=2**33+1)	# tests use of long type for nobj and data
		G.add_edge('A', 1, data='C')
		G.add_edge(1, True, data=3)
		G.add_edge(True, 2**33, data=False)
		G.add_edge(2**33, 'A', 2**33+2) 
		gml.write(G, test_file)
		G_copy = gml.read(test_file)
		self.assertEqual(type(G_copy), Graph)
		self.assertEqual(len(G_copy), 4)
		self.assertEqual(G_copy.size(), 4)
		
		self.assertTrue(G_copy.edge_data('A', 1), 'C')		# see note below**
		self.assertTrue(G_copy.edge_data(1, True), 3)		
		self.assertTrue(G_copy.edge_data(True, str(2**33)), False)
		self.assertTrue(G_copy.edge_data(2**33, 'A'), 2**33+2) 
		
		self.assertEqual(G_copy.node_data('A')['zen_data'], 'B')	# see note below**			
		self.assertEqual(G_copy.node_data(1)['zen_data'], 2)				
		self.assertEqual(G_copy.node_data(True)['zen_data'], False)			
		self.assertEqual(G_copy.node_data(2**33)['zen_data'], 2**33+2)
		
		# these tests all fail because read / write causes bool -> str(bool) and numeric -> str(numeric)
		# it will be necessary to add evaluation to gml.read() which is the intention
		os.remove(test_file)
		
		
	def test_graph_weight(self):
		test_file = path.dirname(__file__) + '/test6.gml'
		G = Graph()
		G.add_node('A')
		G.add_node('B')
		G.add_node('C')
		G.add_edge('A', 'B', weight=0)
		G.add_edge('B', 'C', weight=10)
		G.add_edge('C', 'A', weight=3.141592)
		gml.write(G, test_file)
		G_copy = gml.read(test_file,weight_fxn=lambda data:data['weight'])
		self.assertEqual(G_copy.weight('A','B'),0)
		self.assertEqual(G_copy.weight('B','C'),10)		
		self.assertEqual(G_copy.weight('C','A'),3.141592)
		os.remove(test_file)
	
	#TODO: test including longs in obj / data
	#TODO: test including internal quotes -- expect exception		
		
	def test_large_graph(self):
		test_file = path.dirname(__file__) + '/test7.gml'
		G = Graph()
		num_nodes = 10#0000
		G.add_nodes(num_nodes)

		for i in range (num_nodes -1): # make a large line
			G.add_edge_(i, i+1)

		gml.write(G, test_file)
		G_copy = gml.read(test_file)
		self.assertEqual(len(G), num_nodes)
		preserved_edges = True

		for i in range(num_nodes - 1):
			preserved_edges = preserved_edges and G.has_edge_(i, i+1)

		self.assertTrue(preserved_edges)
		self.assertEqual(G.size(), num_nodes - 1)
		os.remove(test_file)
		

if __name__ == '__main__':
	unittest.main()