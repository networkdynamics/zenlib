import zen
import zen.io.memlist as memlist
import unittest
import os
import os.path as path
import tempfile

class MemListTestCase(unittest.TestCase):

	def test_write_compact_undirected(self):
		G = zen.Graph()
		G.add_nodes(5)
		e1 = G.add_edge_(0,1)
		G.add_edge_(1,2)
		
		# this should succeed
		fd,fname = tempfile.mkstemp()
		os.close(fd)
		memlist.write(G,fname)
		
		# this should fail because the graph isn't compact
		G.rm_edge_(e1)
		try:
			memlist.write(G,fname)
			self.fail('Writing an uncompact graph should raise an exception')
		except zen.ZenException:
			pass
			
	def test_write_compact_directed(self):
		G = zen.DiGraph()
		G.add_nodes(5)
		e1 = G.add_edge_(0,1)
		G.add_edge_(1,2)

		# this should succeed
		fd,fname = tempfile.mkstemp()
		os.close(fd)
		memlist.write(G,fname)

		# this should fail because the graph isn't compact
		G.rm_edge_(e1)
		try:
			memlist.write(G,fname)
			self.fail('Writing an uncompact graph should raise an exception')
		except zen.ZenException:
			pass

	def test_write_undirected(self):
		
		G = zen.Graph()
		G.add_nodes(5)
		G.add_edge_(0,1)
		G.add_edge_(1,2)
		G.add_edge_(3,4)
		
		fd,fname = tempfile.mkstemp()
		os.close(fd)
		memlist.write(G,fname)
		
		G2 = memlist.read(fname,directed=False)
		
		self.assertEqual(len(G2),len(G))
		self.assertEqual(G2.size(),G.size())
		
	def test_write_directed(self):

		G = zen.DiGraph()
		G.add_nodes(5)
		G.add_edge_(0,1)
		G.add_edge_(1,2)
		G.add_edge_(3,4)

		fd,fname = tempfile.mkstemp()
		os.close(fd)
		memlist.write(G,fname)

		G2 = memlist.read(fname,directed=True)

		self.assertTrue(G2.is_directed())
		
		self.assertEqual(len(G2),len(G))
		self.assertEqual(G2.size(),G.size())

	def test_read_directed(self):
		"""
		Test to make sure the last character of the last line is read when a file does not
		end with a \n character.
		"""
		fname = path.join(path.dirname(__file__),'test2.memlist')
		G = memlist.read(fname,directed=True)
		
		self.assertEqual(G.size(),4)
		
		# Last line was read correctly. Don't know how to test for existence of nodes
		# since G.node_object will fail due to lack of node objects.
		self.assertTrue(G.has_edge_(2,3)) # Will raise Exception otherwise.
		
	def test_read_weighted(self):
		"""
		Test to verify if weights are being properly added to the graph.
		Copy of test1.elist with floating point weights between -125000 and 125000.
		"""
		fname = path.join(path.dirname(__file__),'test2_w.memlist')
		G = memlist.read(fname,directed=True,ignore_duplicate_edges=True,weighted=True)

		self.assertEqual(len(G),4)
		self.assertEqual(G.size(),4)

		self.assertEqual(G.weight_(G.edge_idx_(0,1)), 28299.769933)
		
if __name__ == '__main__':
	unittest.main()
