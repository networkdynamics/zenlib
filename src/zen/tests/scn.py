import unittest
from zen import *
import os.path as path
import os

class SCNTestCase(unittest.TestCase):
	
	def test_test1(self):
		G = scn.read(path.join(path.dirname(__file__),'test1.scn'),directed=True)
		
		self.assertEqual(len(G),5)
		self.assertEqual(G.size(),5)
		
		self.assertNotEqual(G.node_data_(0),None)
		self.assertEqual(G.node_data_(0)[0],'1')

	def test_test2(self):
		G = scn.read(path.join(path.dirname(__file__),'test2.scn'),directed=True)
		
		self.assertEqual(len(G),9)
		self.assertEqual(G.size(),0)
		
		self.assertEqual(G.node_data_(0)[0],'E')
	
	def test_test3(self):
		# tests # comments in .scn files
		G = scn.read(path.join(path.dirname(__file__),'test3.scn'), directed=True)
		
		self.assertEqual(len(G),5)
		self.assertEqual(G.size(),5)

		self.assertEqual(G.node_data_(0)[0], '1')
		self.assertEqual(G.node_data_(1)[0], '2')
		self.assertEqual(G.node_data_(2)[0], '3')
		self.assertEqual(G.node_data_(3)[0], '4')
		self.assertEqual(G.node_data_(4)[0], '5')

		self.assertEquals(G.edge_data_(0)[0], '10')
		self.assertEquals(G.edge_data_(1)[0], '11')
		self.assertEquals(G.edge_data_(2)[0], '12')
		self.assertEquals(G.edge_data_(3)[0], '13')
		self.assertEquals(G.edge_data_(4)[0], '14')

	def test_write1(self):
		fname = '___testwrite1__99001324.scn'
		G = scn.read(path.join(path.dirname(__file__),'test1.scn'),directed=True)
		
		self.assertEqual(G.node_data('a')[0],'1')
		self.assertEqual(G.node_data('a')[1],'1')
		
		# write all attributes back
		scn.write(	G,fname,num_node_props=2,num_edge_props=2,
					node_data_fxn=lambda idx,nobj,data: None if data == None else tuple([a for a in data]),
					edge_data_fxn=lambda idx,n1,n2,data: tuple([a for a in data]))
		G = scn.read(fname,directed=True)
		
		self.assertEqual(len(G),5)
		self.assertEqual(G.size(),5)
		
		self.assertNotEqual(G.node_data('a'),None)
		self.assertEqual(G.node_data('a')[0],'1')
		self.assertEqual(G.edge_data('a','b')[0],'X')
		
		# write with no edge attributes
		G = scn.read(path.join(path.dirname(__file__),'test1.scn'),directed=True)
		scn.write(	G,fname,num_node_props=2,num_edge_props=2,
					node_data_fxn=lambda idx,nobj,data: None if data == None else tuple([a for a in data]),
					edge_data_fxn=lambda idx,n1,n2,data: None)
		G = scn.read(fname,directed=True)
		
		self.assertEqual(len(G),5)
		self.assertEqual(G.size(),5)
		
		self.assertNotEqual(G.node_data('a'),None)
		self.assertEqual(G.node_data('a')[0],'1')
		self.assertEqual(G.edge_data('a','b'),None)
		
		os.remove(fname)

	def test_write2(self):
	
		fname = '___testwrite2__99001325.scn'

		G = Graph()
		
		G.add_node('a', '10')
		G.add_node('b', '20')
		G.add_node(1, '30')
		G.add_node('d', '40')

		G.add_edge('a', 'b', '100')
		G.add_edge('b', 1, '0')
		G.add_edge('d', 'a', '10')

		node_data_fnc = lambda idx,nobj,data: None if data == None else [data]
		edge_data_fnc = lambda idx,n1,n2,data: [data]

		scn.write(G,fname,num_node_props=1,num_edge_props=2, node_data_fxn=node_data_fnc, edge_data_fxn=edge_data_fnc)

		os.remove(fname)


if __name__ == "__main__":
	    unittest.main()  
