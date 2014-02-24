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

		self.assertEqual(G.node_data('N1')['sampleOne'],42)
		self.assertEqual(G.node_data('N2')['sampleTwo'],42.1)
		self.assertEqual(G.node_data('N3')['sampleThree'],'HELLO WORLD')

		self.assertEqual(G.edge_data('N1','N2')['label'],
				'Edge from node 1 to node 2')


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
		
		self.assertEqual(G.node_data('N1')['sampleOne'],42)
		self.assertEqual(G.node_data('N2')['sampleTwo'],42.1)
		self.assertEqual(G.node_data('N3')['sampleThree'],'HELLO WORLD')
		
		self.assertEqual(G.edge_data('N1','N2')['label'],
			'Edge from node 1 to node 2')


	def test_list_variables(self):
		fname = path.join(path.dirname(__file__),'test3.gml')
		G = gml.read(fname)
		
		self.assertEqual(len(G),3)
		self.assertEqual(G.size(),2)
		
		self.assertEqual(G.node_data('N1')['listVar'],
			[1,'a',3.2])


	def test_weight_fxn(self):
		fname = path.join(path.dirname(__file__),'test3.gml')
		G = gml.read(fname,weight_fxn=lambda data:data['value'])
		
		self.assertEqual(len(G),3)
		self.assertEqual(G.size(),2)
		
		self.assertEqual(G.weight('N1','N2'),2)
		self.assertEqual(G.weight('N2','N3'),3)


	def test_non_asci_char(self):
		G = Graph()

		G.add_node(u'\u2660')
		G.add_node(u'\u2663')
		G.add_node(u'\u2665')
		G.add_node(u'\u2666')

		G.add_edge(u'\u2663', u'\u2665')
		G.add_edge(u'\u2660', u'\u2666')
		G.add_edge(u'\u2665', u'\u2666')
		G.add_edge(u'\u2660', u'\u2663')

		gml.write(G, 'test4.gml')

		H = gml.read('test4.gml')

		for nobj in G.nodes():
			self.assertEqual(H.node_idx(nobj), G.node_idx(nobj))

		for nobj1, nobj2 in G.edges():
			self.assertEqual(H.edge_idx(nobj1, nobj2), 
				G.edge_idx(nobj1, nobj2))

		self.assertEqual(G.size(), H.size())
		self.assertEqual(len(G), len(H))


	def test_tuple_node_objects(self):
		G = Graph()

		G.add_node((1,2))
		G.add_node((2,3))
		G.add_edge((1,2),(2,3))

		gml.write(G, 'test5.gml')
		H = gml.read('test5.gml')

		for nobj in G.nodes():
			self.assertEqual(H.node_idx(nobj), G.node_idx(nobj))

		for nobj1, nobj2 in G.edges():
			self.assertEqual(H.edge_idx(nobj1, nobj2), 
				G.edge_idx(nobj1, nobj2))

		self.assertEqual(G.size(), H.size())
		self.assertEqual(len(G), len(H))


	def test_no_node_data(self):
		G = Graph()
		G.add_node()
		G.add_node()
		G.add_edge_(0,1)

		gml.write(G, 'test5.gml')
		H = gml.read('test5.gml')

		for edge_idx in G.edges_():
			node_idx1, node_idx2 = H.endpoints_(edge_idx)
			H.has_edge_(node_idx1, node_idx2)

		self.assertEqual(G.size(), H.size())
		self.assertEqual(len(G), len(H))


if __name__ == '__main__':
	unittest.main()
