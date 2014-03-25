import unittest
import random
import types
import pickle

from zen import *

class GraphBuildFromTestCase(unittest.TestCase):

	def test_bad_argument(self):
		import numpy as np

		A = np.ones((10,10))
		
		with self.assertRaises(ValueError):
			Graph.from_adj_matrix(A,bad_argument=3)

	def test_ndarray_ekman(self):
		import numpy as np

		A = np.ones((10,10))
		G = Graph.from_adj_matrix(A)

		self.assertEquals(len(G),10)
		self.assertEquals(G.size(),55)

		# done

	def test_ndarray_ekman2(self):
		import numpy as np

		A = np.ones((10,10))
		G = Graph.from_adj_matrix(A,node_obj_fxn=None)
		
		self.assertEquals(len(G),10)
		self.assertEquals(G.size(),55)

		with self.assertRaises(ZenException):
			G.nodes()

		# done


class GraphCompact(unittest.TestCase):
	
	def test_compact_nodes(self):
		G = Graph()
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(3,4)
		G.rm_node(2)
		
		self.assertEquals(G.max_node_idx,3)
		
		G.compact()
		
		self.assertEquals(G.max_node_idx,2)
		self.assertEquals(G.node_idx(1),0)
		self.assertEquals(G.node_idx(3),2)
		self.assertEquals(G.node_idx(4),1)
		self.assertEquals(G.edge_idx(3,4),0)
		
		G.validate()
	
	def test_compact_selfedges(self):
		G = Graph()
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(3,3)
		
		self.assertEquals(G.degree(3),3)
		
		G.rm_node(2)
		
		G.compact()
		
		self.assertEquals(G.max_node_idx,1)
		self.assertEquals(G.node_idx(3),1)
		self.assertEquals(G.edge_idx(3,3),0)
		self.assertEquals(G.degree(3),2)
		
		G.validate()
		
	def test_compact_selfedges2(self):
		"""
		I think there was a problem with how the edge indices were being set internally during graph compacting...
		"""
		G = Graph()
		G.add_edge(1,2)
		G.add_edge(2,4)
		G.add_edge(4,5)
		G.add_edge(4,3)
		G.add_edge(2,3)
		G.add_edge(5,3)
		G.add_edge(3,3)
		
		self.assertEquals(G.degree(3),5)
		
		G.rm_node(2)
		
		self.assertEquals(G.degree(3),4)
		
		G.compact()
		
		self.assertEquals(G.max_node_idx,3)
		self.assertEquals(G.node_idx(3),1)
		self.assertEquals(G.edge_idx(3,3),0)
		self.assertEquals(G.degree(3),4)
		
		self.assertTrue(G.is_compact())
		
		G.rm_edge(3,3)
		
		self.assertEquals(G.degree(3),2)
		
		G.validate()
	
	def test_compact_fail_nodes(self):
		G = Graph()
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(3,3)
		G.rm_node(1)
		G.rm_node(2)
		
		G.compact()
		
		G.validate()
		
	def test_compact_edges(self):
		G = Graph()
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.rm_edge(1,2)
	
		self.assertEquals(G.max_edge_idx,1)
		
		G.compact()
		
		self.assertEquals(G.max_edge_idx,0)
		
		G.validate()
	
	def test_is_compact(self):
		G = Graph()
		G.add_edge(1,2)
		G.add_edge(2,3)
		
		self.assertTrue(G.is_compact())
		
		G.rm_node(2)
		
		self.assertFalse(G.is_compact())
		
		G.add_edge(3,4)
		G.add_edge(1,4)
		
		self.assertTrue(G.is_compact())
		
		G.validate()
		
class GraphAddXFunctions(unittest.TestCase):
	
	def test_add_node_x_error(self):
		G = Graph()
		
		# put a node into the graph
		idx = G.add_node()
		
		# try to overwrite that node
		try:
			G.add_node_x(idx,G.edge_list_capacity,None,None)
			self.fail('Attempt to overwrite node using G.add_node_x succeeded')
		except ZenException:
			pass
			
		G.validate()
			
	def test_add_edge_x_error(self):
		G = Graph()

		# put a node into the graph
		G.add_node(1)
		G.add_node(2)
		idx = G.add_edge(1,2)
		G.add_node(3)

		# try to overwrite that node
		try:
			G.add_edge_x(idx,1,2,None,1)
			self.fail('Attempt to overwrite edge using G.add_edge_x succeeded')
		except ZenException:
			pass
			
		G.validate()
			
	def test_nodes_after_add_node_x(self):

		graph = Graph()
		graph.add_node_x(0, 10, 0, None)
		graph.add_node_x(1, 10, 1, None)
		graph.add_node_x(2, 10, 2, None)

		nodes = graph.nodes_()
		self.assertEquals(0, nodes[0])
		self.assertEquals(1, nodes[1])
		self.assertEquals(2, nodes[2])
		
		graph.validate()
		
	def test_free_node_list(self):
		"""
		If a node is added via add_node_x, any nodes
		that are skipped over should be used when other
		nodes are added later.
		"""
		G = Graph()
		G.add_node_x(1,10, None, None)
		
		idx = G.add_node()
		
		self.assertEqual(idx,0)
		
		G.validate()
		
	def test_free_edge_list(self):
		"""
		If an edge is added via add_edge_x, any edge
		that are skipped over should be used when other
		edges are added later.
		"""
		G = Graph()
		G.add_node(1)
		G.add_node(2)
		G.add_node(3)
	
		G.add_edge_x(1,0,1,None,1)
	
		idx = G.add_edge(2,3)
	
		self.assertEqual(idx,0)
		
		G.validate()
			
class GraphRelabelTestCase(unittest.TestCase):
	
	def test_change_node_obj(self):
		G = Graph()
		G.add_node(1,data=1)
		G.set_node_object(1,2)
		
		self.assertFalse(1 in G)
		self.assertTrue(2 in G)
		
		G.validate()

class GraphCopyTestCase(unittest.TestCase):
	
	def test_basic_index_preservation(self):
		G = Graph()
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(1,3)
		G.rm_node(2)
		
		G2 = G.copy()
		
		self.assertEqual(G.node_idx(1),G2.node_idx(1))
		self.assertEqual(G.node_idx(3),G2.node_idx(3))
		self.assertEqual(G.edge_idx(1,3),G2.edge_idx(1,3))
		self.assertEqual(G.edge_idx(1,3),G2.edge_idx(3,1))
		
		G.validate()
		G2.validate()

class GraphPickleTestCase(unittest.TestCase):
	
	def test_basic(self):
		G = Graph()
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(3,1)
		
		pstr = pickle.dumps(G)
		
		G2 = pickle.loads(pstr)
		assert G2.has_edge(1,2)
		assert G2.has_edge(2,3)
		assert G2.has_edge(3,1)
		assert G2.has_edge(1,3)
		
		G.validate()
		
	def test_removal(self):
		G = Graph()
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(3,4)
		G.add_edge(4,5)
		G.rm_node(3)
		
		idx2 = G.node_idx(2)
		eidx45 = G.edge_idx(4,5)
		
		pstr = pickle.dumps(G)
		
		G2 = pickle.loads(pstr)
		assert len(G2) == 4
		assert G2.size() == 2
		assert G.node_idx(2) == idx2
		assert G.edge_idx(4,5) == eidx45
		
		# make sure things don't crash when we add a new node - are internal structures valid?
		G2.add_edge(5,6)
		
		G.validate()

class GraphTestCase(unittest.TestCase):
		
	def test_add_bad_nobjs(self):
		G = Graph()
		G.add_node("x")
		
		try:
			G.add_node("x")
			self.fail('adding a second node with the same node object should have raised an exception')
		except ZenException:
			pass
		
	def test_add_nodes(self):
		G = Graph()
		G.add_nodes(10)
		
		self.assertEquals(len(G),10)
		
		G.validate()
		
	def test_add_nodes_w_objects(self):
		G = Graph()
		G.add_nodes(10,lambda x:str(x))
		
		self.assertEquals(len(G),10)
		
		for n_,n in G.nodes_iter_(obj=True):
			self.assertEquals(str(n_),n)
			
		G.validate()
					
	def test_adj_matrix(self):
		G = Graph()
		G.add_edge(0,1)
		G.add_edge(1,2)
		G.add_edge(2,2)
		
		M = G.matrix()
		inf = float('infinity')
		self.assertEquals(M[0,0],0)
		self.assertEquals(M[0,1],1)
		self.assertEquals(M[0,2],0)
		self.assertEquals(M[1,0],1)
		self.assertEquals(M[1,1],0)
		self.assertEquals(M[1,2],1)
		self.assertEquals(M[2,0],0)
		self.assertEquals(M[2,1],1)
		self.assertEquals(M[2,2],1)
		
		G.validate()
	
	def test_node_removal_nodes_(self):
		
		graph = Graph()
		graph.add_edge(1,2)
		graph.add_edge(3,2)
		graph.add_edge(3,4)
		graph.add_edge(3,6)
		nset1 = set(graph.nodes_())
		graph.rm_node_(2)
		nset2 = set(graph.nodes_())
		
		nset1.remove(2)
		
		self.assertTrue(4 in nset2)
		self.assertEqual(nset1,nset2)
		
		graph.validate()
	
	def test_max_indices(self):
		G = Graph()
		n1 = G.add_node()
		n2 = G.add_node()
		n3 = G.add_node()
		
		G.add_edge_(n1,n2)
		G.add_edge_(n2,n3)
		
		self.assertEqual(G.max_node_idx,2)
		self.assertEqual(G.max_edge_idx,1)
		
		G.validate()
	
	def test_recycle_node_indices(self):
		G = Graph(node_capacity=5)
		for i in range(5):
			G.add_node(i)
		
		G.validate()
		
		for i in range(5):
			G.rm_node(i)
		
		G.validate()
		
		for i in range(5):
			G.add_node(i)
		
		G.validate()
		
		self.assertEqual(G.node_capacity,5)
		
	def test_recycle_nodes_beyond_next_node_idx(self):
		"""
		Here we test the ability for a graph to keep nodes in
		its free list that are beyond the next_node_idx
		"""
		pass
	
	def test_recycle_edge_indices(self):
		G = Graph(edge_capacity=5)

		G.add_edge(1,2)
		G.add_edge(1,3)
		G.add_edge(1,4)
		G.add_edge(1,5)
		G.add_edge(1,6)

		G.rm_edge(1,2)
		G.rm_edge(1,3)
		G.rm_edge(1,4)
		G.rm_edge(1,5)
		G.rm_edge(1,6)

		G.add_edge(1,2)
		G.add_edge(1,3)
		G.add_edge(1,4)
		G.add_edge(1,5)
		G.add_edge(1,6)

		self.assertEqual(G.edge_capacity,5)	
		
		G.validate()
	
	def test_tutorial1(self):
		G = Graph()
		
		self.assertEqual(len(G),0)
		self.assertEqual(G.size(),0)
		
		for i in range(100):
			G.add_node(i)
			
		self.assertEqual(len(G),100)
		
		i = 0
		while i < 100:
			x = random.randint(0,99)
			y = random.randint(0,99)
			if not G.has_edge(x,y):
				G.add_edge(x,y)
				i += 1
				
		self.assertEqual(G.size(),100)
		G.add_edge('hello','world')
		self.assertEqual(G.size(),101)
		self.assertEqual(len(G),102)
		
		degrees = [G.degree(n) for n in G.nodes_iter()]
		degrees.sort()
		
		# import pylab as pl
		# pl.figure()
		# pl.plot(degrees)
		# pl.show()
				
		for u,v in G.edges_iter():
			G.set_weight(u,v,random.random())
			
		for n in G.nodes_iter():
			max_weight = 0
			max_edge = None
			for u,v,w in G.edges_iter(n,weight=True):
				if w > max_weight:
					max_weight = w
					max_edge = (u,v)

			if max_edge is None:
				#print 'Node %d has no neighbors' % n
				pass
			else:
				n2 = max_edge[0]
				if n2 == n:
					n2 = max_edge[1]
					#print "Node %d's heaviest neighbor is %d" % (n,n2)
		
		G.validate()
		
		return
	
	def test_basicadding(self):
		G = Graph()
		n1 = G.add_node('hello')
		n2 = G.add_node('there')
		n3 = G.add_node('world')
		
		self.assertEqual(len(G),3)
		self.assertEqual(G.node_object(n1),'hello')
		self.assertEqual(G.node_object(n2),'there')
		self.assertEqual(G.node_object(n3),'world')
		self.assertEqual(G.node_data_(n3),None)
		
		# add by ids
		e1 = G.add_edge_(n1,n2)
		
		self.assertTrue(G.has_edge_(n1,n2))
		self.assertTrue(G.has_edge_(n2,n1))
		self.assertEqual(G.edge_idx_(n1,n2),e1)
		self.assertFalse(G.has_edge_(n1,n3))
		self.assertEqual(G.edge_data_(G.edge_idx_(n1,n2)),None)
		
		# check containment
		self.assertTrue('hello' in G)
		self.assertFalse('haloo' in G)
		
		# add by names
		e2 = G.add_edge('there','world')
		x,y = G.endpoints_(e2)
		
		self.assertEqual(x,n2)
		self.assertEqual(y,n3)
		self.assertTrue(G.has_edge_(n2,n3))
		self.assertEqual(G.edge_idx_(n2,n3),e2)
		self.assertEqual(G.edge_idx_(n2,n3),G.edge_idx('there','world'))
		
		# test degree
		self.assertEqual(G.degree_(n1),1)
		self.assertEqual(G.degree_(n2),2)
		
		G.validate()
	
	def test_nodes(self):
		G = Graph()
		n1 = G.add_node('hello')
		G.add_edge('hello','there')
		G.add_edge('hello','world')
		
		self.assertEqual(type(G.nodes()),types.ListType)
		self.assertEqual(type(G.neighbors('hello')),types.ListType)
		self.assertEqual(len(G.neighbors('hello')),2)
		
		G.validate()
	
	def test_nodes_(self):
		G = Graph()
		n1 = G.add_node('hello')
		G.add_edge('hello','there')
		G.add_edge('there','world')
	
		self.assertEqual(len(G.neighbors_(n1)),1)
		
		G.validate()
	
	def test_edge_removal(self):
		G = Graph()
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(3,4)
		
		self.assertEqual(G.size(),3)
		G.rm_edge(2,3)
		self.assertEqual(G.size(),2)
		
		G.validate()
	
	def test_duplicate_edges(self):
		G = Graph()
		G.add_edge('1','2')
		
		# add it one way
		success = False
		try:
			G.add_edge('1','2')
			success = True
		except ZenException,e:
			if not str(e).startswith('Duplicate edges'):
				self.fail('Incorrect exception: %s' % str(e))
			
		self.assertFalse(success,'This call should have thrown an exception')
		
		G.validate()
		
		# add it the other way
		success = False
		try:
			G.add_edge('2','1')
			success = True
		except ZenException,e:
			if not str(e).startswith('Duplicate edges'):
				self.fail('Incorrect exception: %s' % str(e))
			
		self.assertFalse(success,'This call should have thrown an exception')
		
		G.validate()
		
	def test_neighbors2(self):
		G = Graph()
		G.add_edge('hello','there')
		
		self.assertTrue('hello' in set(G.neighbors_iter('there')))
		
	def test_small_edge_insertion(self):
		G = Graph()
		
		for i in range(10):
			G.add_node()
		
		order = range(10)	
		order.remove(5)
		random.shuffle(order)
		
		for x in order:
			G.add_edge_(5,x)
			self.assertTrue(G.has_edge_(5,x))
			
		random.shuffle(order)
		for x in range(10):
			if x != 5:
				self.assertTrue(G.has_edge_(x,5))
				self.assertTrue(G.has_edge_(5,x))
				
		G.validate()

	def test_growing_nodearray(self):
		G = Graph(node_capacity=1)
		for i in range(10000):
			n = G.add_node(i)
			
		G.validate()
			
	def test_growing_edgelistarray(self):
		G = Graph(edge_list_capacity=1)
		n0 = G.add_node('hello')
	
		for i in range(1000):
			n = G.add_node(i)
			G.add_edge_(n0,n)
			
		G.validate()
		
	def test_rm_node_edge(self):
		G = Graph()
		n1 = G.add_node()
		n2 = G.add_node('hello')
		n3 = G.add_node('foobar')
		
		e1 = G.add_edge_(n1,n2)
		e2 = G.add_edge_(n2,n3)
		e3 = G.add_edge_(n3,n1)
		
		G.rm_edge_(e3)
		
		self.assertFalse(G.has_edge_(n3,n1))
		self.assertTrue(G.has_edge_(n1,n2))
		self.assertTrue(G.has_edge_(n2,n3))
		self.assertTrue(G.has_edge_(n3,n2))
		
		self.assertEqual(G.degree_(n1),1)
		self.assertEqual(G.degree_(n2),2)
		
		G.rm_node_(n1)
		
		self.assertEqual(G.degree_(n2),1)
		self.assertEqual(G.degree_(n3),1)
		
		G.validate()
		
	def test_node_iterator(self):
		G = Graph()
		for i in range(1000):
			G.add_node(i)
		
		count = 0
		for i,d in G.nodes_iter(data=True):
			count += 1
			
		self.assertEqual(count,1000)
		
		G.validate()
		
	def test_edge_iterator(self):
		G = Graph()
		n1 = G.add_node()
		for i in range(1000):
			n = G.add_node()
			G.add_edge(n1,n)
			
		count = 0
		for x,y,d in G.edges_iter(data=True):
			count += 1
			
		self.assertEqual(count,1000)
		
		G.validate()
		
	def test_edges(self):
		G = Graph()
		
		G.add_edge('x','y')
		E = G.edges()
		e1 = E[0]
		
		self.assertTrue('x' in set(e1))
		
		G.validate()
	
	def test_grp_edge_iterators(self):
		G = Graph()
		n1 = G.add_node('n1')
		n2 = G.add_node()
		G.add_edge_(n1,n2)
		for i in range(10):
			n = G.add_node(i)
			G.add_edge_(n,n1)
			G.add_edge_(n2,n)
		
		# test total group iterators
		count = 0
		for eid in G.grp_edges_iter_([n1,n2]):
			count += 1
		
		self.assertEqual(count,21)
		
		# test the return of endpoints
		success = True
		try:
			for x,y in G.grp_edges_iter(['n1']):
				self.assertTrue(y == 'n1' or x == 'n1')
		except ZenException:
			success = False
			
		self.assertFalse(success)
		
		G.validate()
		
	def test_grp_neighbor_iterators(self):
		G = Graph()
		n1 = G.add_node()
		n2 = G.add_node()
		G.add_edge_(n1,n2)
		for i in range(10):
			n = G.add_node()
			G.add_edge_(n,n1)
			G.add_edge_(n2,n)
	
		# test total group iterators
		count = 0
		for eid in G.grp_neighbors_iter_([n1,n2]):
			count += 1
	
		self.assertEqual(count,12)
		
		G.validate()
			
	def test_neighbor_iters(self):
		G = Graph()
		n1 = G.add_node('n1')
		n2 = G.add_node('n2')
		n3 = G.add_node('n3')
		n4 = G.add_node('n4')
		
		G.add_edge_(n1,n2)
		G.add_edge_(n1,n3)
		G.add_edge_(n4,n1)
		
		n1_all_raw = list(G.neighbors_iter_(n1))
		n1_all = set(n1_all_raw)
		
		self.assertEquals(n1_all,set([n2,n3,n4]))
		self.assertEquals(len(n1_all_raw),3)
		
		G.validate()
		
	def test_neighbor_iter_recursionlimit(self):
		"""
		This tests an incorrect way that neighbor iter was implemented initially.
		"""
		G = Graph()
		n1 = G.add_node()
		for i in range(1100):
			n = G.add_node()
			G.add_edge_(n,n1)
			
		count = 0
		for n in G.neighbors_iter_(n1):
			count += 1
			
		self.assertEquals(count,1100)
		
		G.validate()
		
	def test_node_iterator_with_obj_and_data(self):
		G = Graph()
		G.add_node()
		G.add_node('there')
		
		for idx,nobj,data in G.nodes_iter_(obj=True,data=True):
			self.assertEqual(data,None)
			
		G.validate()
			
	def test_edge_iterator_with_obj_and_data(self):
		G = Graph()
		G.add_edge(1,2)
		G.add_edge(2,3)

		for eidx,data in G.edges_iter_(data=True):
			self.assertEqual(data,None)
			
		G.validate()
			
	def test_neighbor_iterator_with_obj_and_data(self):
		G = Graph()
		G.add_edge(1,2)
		G.add_edge(1,3)

		for x,nobj,data in G.neighbors_iter_(G.node_idx(1),obj=True,data=True):
			self.assertEqual(data,None)
			
		G.validate()
			
	def test_neighbor_iterator_with_data(self):
		G = Graph()
		n1 = G.add_node('1')
		n2 = G.add_node()
		n3 = G.add_node('2')
		G.add_edge_(n1,n2)
		G.add_edge(n1,n3)

		success = True
		try:
			for nobj,data in G.neighbors_iter('1',data=True):
				self.assertEqual(data,None)
		except ZenException, e:
			success = False
			self.assertTrue('does not have a node object' in str(e))
	
		self.assertFalse(success)
		
		G.validate()
		
	def test_set_data(self):
		G = Graph()
		G.add_node(1,'hello')
		G.add_edge(1,2,'x')

		G.set_node_data(1,'there')
		G.set_edge_data(2,1,'y')

		self.assertEqual(G.node_data(1),'there')
		self.assertEqual(G.edge_data(1,2),'y')
		
		G.validate()
		
	def test_weights(self):
		G = Graph()
		G.add_edge(1,2,weight=2)
		G.add_edge(1,3)
		G.set_weight(1,3,5)
		
		self.assertEqual(G.weight(1,2),2)
		self.assertEqual(G.weight(1,3),5)
		
		G.validate()
		
	def test_edge_data(self):
		G = Graph()
		G.add_edge(1,2,1)
		G.add_edge(2,3)

		for e,data in G.edges_(data=True):
			pass

		for x,y,data in G.edges(data=True):
			pass

		n1 = G.add_node()
		n2 = G.add_node()
		G.add_edge_(n1,n2)

		success = True
		try:
			for x,y,data in G.edges(data=True):
				pass
		except ZenException, e:
			if str(e).startswith('Edge'):
				success = False

		self.assertFalse(success)
		
		G.validate()
		
	def test_modify_node_iterator(self):
		G = Graph()
		G.add_node(0)
		for i in range(1,100):
			G.add_node(i)
			G.add_edge(i,i-1)
				
		error = False
		try:	
			for n in G.nodes_iter():
				G.rm_node(n)
		except GraphChangedException:
			error = True
		
		self.assertTrue(error)
		
		G.validate()
		
	def test_modify_edge_iterator(self):
		G = Graph()
		G.add_node(0)
		for i in range(1,100):
			G.add_node(i)
			G.add_edge(i,i-1)
				
		error = False
		try:	
			for n1,n2 in G.edges_iter():
				G.rm_edge(n1,n2)
		except GraphChangedException:
			error = True
		
		self.assertTrue(error)
		
		G.validate()
		
	def test_modify_neighbor_iterator(self):
		G = Graph()
		G.add_node(0)
		for i in range(1,100):
			G.add_node(i)
			G.add_edge(0,i)

		error = False
		try:	
			for n in G.neighbors_iter(0):
				G.rm_node(n)
		except GraphChangedException:
			error = True

		self.assertTrue(error)
		
		G.validate()
		
	def test_invalid_nidx(self):
		G = Graph()
		n1 = G.add_node(0)

		error = False
		try:
			G.node_data_(n1+1)
		except ZenException:
			error = True

		self.assertTrue(error)
		
		G.validate()
		
	def test_edge_idx(self):
		G = Graph()
		n1 = G.add_node()
		n2 = G.add_node()
		e1 = G.add_edge_(n1,n2,'blah')

		self.assertEquals(e1,G.edge_idx_(n1,n2))
		
		G.validate()
		
	def test_has_edge(self):
		G = Graph()

		try:
			r = G.has_edge(1,2)
			self.assertFalse(r)
		except:
			self.fail('No error should be thrown')
		
		G.validate()
		
	def test_edges_no_data(self):
		G = Graph()
		G.add_edge(1,2,data=None)

		for e in G.edges_(G.node_idx(1),data=True):
			pass

		# if we made it here, we won

if __name__ == '__main__':
	unittest.main()
