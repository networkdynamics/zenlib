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
			
	# 	def test_load_zen_data(self):
	# 		fname = path.dirname(__file__) + '/test4.gml'
	# 		G1 = gml.read(fname)
	# 		
	# 		self.AssertEqual(G1.node_data_(0), {'label':'soft', 'attr1':'one', 'attr2':'two'})
	# 		self.AssertEqual(G1.node_data_(1), {'label':'soft', 'attr1':'one', 'attr2':'two'})
	# 		self.AssertEqual(G1.edge_data_(0,1), {'label':'soft', 'attr1':'one', 'attr2':'two'})
	# 		self.AssertEqual(G1.edge_data_(0,1), {'zenData':'included if other data present','label':'soft', 'attr1':'one', 'attr2':'two'})
		


class GMLWriteTestCase(unittest.TestCase):
		
	def test_empty_graph(self):
		test_file = path.dirname(__file__) + '/test5.gml'
		G = Graph()
		gml.write(G, test_file)
		G_ = gml.read(test_file)
		self.assertEqual(type(G),Graph)
		self.assertEqual(len(G),0)
		self.assertEqual(G.size(),0)
		os.remove(test_file)
		
	def test_empty_digraph(self):	
		test_file = path.dirname(__file__) + '/test6.gml'
		G = DiGraph()
		gml.write(G, test_file)
		G_ = gml.read(test_file)
		self.assertEqual(type(G),DiGraph)
		self.assertEqual(len(G),0)
		self.assertEqual(G.size(),0)
		os.remove(test_file)
		
	def test_empty_BipartiteGraph(self):
		test_file = path.dirname(__file__) + '/test7.gml'
		G = BipartiteGraph()
		gml.write(G, test_file)
		G_copy = gml.read(test_file)
		# TODO: consider recognizing a bipartite parameter during read
		# self.assertEqual(type(G_copy),BipartiteGraph)
		self.assertEqual(len(G_copy),0)
		self.assertEqual(G_copy.size(),0)
		#os.remove(test_file)
	
	# TODO: test write_data and use_zen_data options to gml.write()
	# this will be easier when gml.read() is updated. 
	
	def test_attributes_preserved(self):
		test_file = path.dirname(__file__) + '/test8.gml'
		
		nested_datum = {'alist':[1,'two', 3.14], 'specialChars':{'xmlSpecialChars':'&<>\\\"', 'nonAsciiChars':'π∂ƒ∆†'}}
		
		G = Graph()
		G.add_node(nobj='A', data=nested_datum)
		G.add_node(nobj='B', data=False)
		G.add_node(nobj='C', data=2**33)
		G.add_edge('A', 'B', data='C', weight=0)
		G.add_edge('B', 'C', data=False, weight=10)
		G.add_edge('C', 'A', data=2**33+1, weight=3.141592) 
		gml.write(G, test_file)
		G_copy = gml.read(test_file)
		
		# Graph type and connectivity preserved
		self.assertEqual(type(G_copy), Graph)
		self.assertEqual(len(G_copy), 3)
		self.assertEqual(G_copy.size(), 3)
		
		# node data preserved
		self.assertEqual(G_copy.node_data('A'), nested_datum)		
		self.assertEqual(G_copy.node_data('B'), False)			
		self.assertEqual(G_copy.node_data('C'), 2**33)
		
		# edge data preserved
		self.assertEqual(G_copy.edge_data('A', 'B'), 'C')		
		self.assertEqual(G_copy.edge_data('B', 'C'), False)
		self.assertEqual(G_copy.edge_data('C', 'A'), 2**33+1) 
		
		# edge weights preserved
		self.assertEqual(G_copy.weight('A','B'),0)
		self.assertEqual(G_copy.weight('B','C'),10)		
		self.assertEqual(G_copy.weight('C','A'),3.141592)
		
		os.remove(test_file)
		
	def test_zen_data(self):
		test_file = path.dirname(__file__)
		G = Graph()
		G.add_node('A', data={'att1':'one','att2':'two'})
		G.add_node('B', data={'att3':'three','att4':'four'})
		G.add_node('C')
		G.add_edge('A','B', data={'att5':'five', 'att6':'six'})
		G.add_edge('B', 'C')
		
		# data is recorded loose in the node dictionaries, and recovered from there.  
		# comment out os.remove line to see in written file.
		gml.write(G, test_file + '/test8.gml', write_data=True, use_zen_data=False) 
		G_1 = gml.read(test_file + '/test8.gml')
		self.assertEqual(G_1.node_data('A'), {'att1':'one','att2':'two'})
		self.assertEqual(G_1.node_data('B'), {'att3':'three','att4':'four'})
		self.assertEqual(G_1.edge_data('A','B'), {'att5':'five', 'att6':'six'})
		os.remove(test_file + '/test8.gml')
		
		gml.write(G, test_file + '/test9.gml', write_data=False)
		G_2 = gml.read(test_file + '/test9.gml')
		self.assertEqual(G_2.node_data('A'), None)
		self.assertEqual(G_2.node_data('B'), None)
		self.assertEqual(G_2.edge_data('A','B'), None)
		os.remove(test_file + '/test9.gml')
				
	def test_large_graph(self):
		test_file = path.dirname(__file__) + '/test10.gml'
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