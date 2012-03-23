from zen import *
import unittest
import os
import os.path as path
import tempfile

class EdgeListWriteTestCase(unittest.TestCase):
	"""
	Tests to write:
		- test_write_undirected1_labels_noweights
		- test_write_directed1_labels_noweights
		- test_write_undirected1_labels_weights
		- test_write_directed1_labels_weights
	"""
	def test_write_undirected1_nolabels_noweights(self):
		G = Graph()
		G.add_edge(0,1)
		G.add_edge(1,2)
		
		fd,fname = tempfile.mkstemp()
		os.close(fd)
		edgelist.write(G,fname)
		
		G = edgelist.read(fname,directed=False)
		
		self.assertTrue(G.has_edge('0','1'))
		self.assertTrue(G.has_edge('1','2'))
		self.assertTrue(G.has_edge('2','1'))
		
		os.remove(fname)
		
	def test_write_undirected1_nolabels_weights(self):
		G = Graph()
		G.add_edge(0,1,weight=5.5)
		G.add_edge(1,2,weight=-2.3)

		fd,fname = tempfile.mkstemp()
		os.close(fd)
		edgelist.write(G,fname,use_weights=True)

		G = edgelist.read(fname,directed=False,weighted=True)

		self.assertTrue(G.has_edge('0','1'))
		self.assertTrue(G.has_edge('1','2'))
		self.assertTrue(G.has_edge('2','1'))
		self.assertEqual(G.weight('0','1'),5.5)
		self.assertEqual(G.weight('2','1'),-2.3)

		os.remove(fname)
		
	def test_write_directed1_nolabels_noweights(self):
		G = DiGraph()
		G.add_edge(0,1)
		G.add_edge(1,2)

		fd,fname = tempfile.mkstemp()
		os.close(fd)
		edgelist.write(G,fname)

		G = edgelist.read(fname,directed=True)

		self.assertTrue(G.has_edge('0','1'))
		self.assertTrue(G.has_edge('1','2'))
		self.assertFalse(G.has_edge('2','1'))

		os.remove(fname)
		
	def test_write_directed1_nolabels_weights(self):
		G = DiGraph()
		G.add_edge(0,1,weight=5.5)
		G.add_edge(1,2,weight=-2.3)

		fd,fname = tempfile.mkstemp()
		os.close(fd)
		edgelist.write(G,fname,use_weights=True)

		G = edgelist.read(fname,directed=True,weighted=True)

		self.assertTrue(G.has_edge('0','1'))
		self.assertTrue(G.has_edge('1','2'))
		self.assertFalse(G.has_edge('2','1'))
		self.assertEqual(G.weight('0','1'),5.5)
		self.assertEqual(G.weight('1','2'),-2.3)

		os.remove(fname)

class EdgeListReadTestCase(unittest.TestCase):

	def test_read_undirected1(self):
		fname = path.join(path.dirname(__file__),'utest1.elist')
		G = edgelist.read(fname,directed=False)
		self.assertTrue(type(G) == Graph)
		
		self.assertEqual(len(G),4)
		self.assertEqual(G.size(),4)
		
		self.assertTrue('0' in G)
		self.assertTrue('1' in G)
		self.assertTrue('2' in G)
		self.assertTrue('3' in G)
		self.assertFalse('4' in G)
		
		self.assertTrue(G.has_edge('0','1'))
		self.assertTrue(G.has_edge('1','2'))
		self.assertTrue(G.has_edge('2','1'))
		self.assertTrue(G.has_edge('2','3'))
		self.assertTrue(G.has_edge('2','3'))
		self.assertTrue(G.has_edge('3','1'))

	def test_read_directed1(self):
		fname = path.join(path.dirname(__file__),'test1.elist')
		G = edgelist.read(fname,directed=True)
		self.assertTrue(type(G) == DiGraph)
		
		self.assertEqual(len(G),396)
		self.assertEqual(G.size(),200)
		
	def test_read_directed2(self):
		"""
		Test to make sure the last character of the last line is read when a file does not
		end with a \n character.
		"""
		fname = path.join(path.dirname(__file__),'test2.elist')
		G = edgelist.read(fname,directed=True)
		self.assertTrue(type(G) == DiGraph)
		
		self.assertTrue('62' in G)
		
	def test_read_directed3(self):
		"""
		Loading this file was causing an incorrect duplicate edge exception.
		"""
		fname = path.join(path.dirname(__file__),'er_10_0.5.elist')
		G = edgelist.read(fname,directed=True)
		self.assertTrue(type(G) == DiGraph)
		
	def test_read_directed_duplicates(self):
		"""
		Test to make sure that duplicates are ignored if
		this is specified.
		"""
		fname = path.join(path.dirname(__file__),'test3.elist')
		G = edgelist.read(fname,directed=True,ignore_duplicate_edges=True)
		self.assertTrue(type(G) == DiGraph)
		
		self.assertTrue('62' in G)
		
	def test_read_directed_weighted(self):
		"""
		Test to verify if weights are being properly added to the graph.
		"""
		fname = path.join(path.dirname(__file__),'test2_w.elist')
		G = edgelist.read(fname,directed=True,weighted=True)
		self.assertTrue(type(G) == DiGraph)
		
		self.assertEqual(len(G),4)
		self.assertEqual(G.size(),4)
		
		self.assertEquals(G.weight('1','2'),0)
		self.assertEquals(G.weight('2','3'),-5)
		self.assertEquals(G.weight('1','3'),2.3)
		self.assertEquals(G.weight('3','62'),-3.2)
		
	def test_read_undirected_weighted(self):
		"""
		Test to verify if weights are being properly added to the graph.
		"""
		fname = path.join(path.dirname(__file__),'test2_w.elist')
		G = edgelist.read(fname,directed=False,weighted=True)
		self.assertTrue(type(G) == Graph)

		self.assertEqual(len(G),4)
		self.assertEqual(G.size(),4)

		self.assertEquals(G.weight('1','2'),0)
		self.assertEquals(G.weight('2','3'),-5)
		self.assertEquals(G.weight('3','2'),-5)
		self.assertEquals(G.weight('1','3'),2.3)
		self.assertEquals(G.weight('3','62'),-3.2)
		self.assertEquals(G.weight('62','3'),-3.2)
		
	def test_read_helist1(self):
		fname = path.join(path.dirname(__file__),'test1.helist')
		G = hedgelist.read(fname)
		
		self.assertEqual(len(G),7)
		self.assertEqual(G.size(),4)
		
if __name__ == '__main__':
	unittest.main()