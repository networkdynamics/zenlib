from zen import *
import unittest
import random

class RandomizeTestCase(unittest.TestCase):
	
	def test_choose_node_bad_arg(self):
		G = Graph()
		G.add_edge(1,2)
		
		try:
			choose_node(G,blah=10)
			self.fail('blah should not be accepted as a keyword argument')
		except ZenException,e:
			pass
	
	def test_choose_node(self):
		NUM_NODES = 20
		FINAL_NUM_NODES = 10
		
		# build the graph
		G = Graph()
		nodes = range(NUM_NODES)
		for i in nodes:
			G.add_node(i)
		
		# remove some nodes
		random.shuffle(nodes)
		rm_nodes = nodes[FINAL_NUM_NODES:]
		nodes = set(nodes[:FINAL_NUM_NODES])
		for n in rm_nodes:
			G.rm_node(n)
		
		# check to see how the node selection goes
		for i in range(NUM_NODES*5):
			self.assertTrue(choose_node(G) in nodes)
			self.assertTrue(G.node_object(choose_node_(G)) in nodes)
	
	def test_choose_edge_bad_arg(self):
		G = Graph()
		G.add_edge(1,2)

		try:
			choose_edge(G,blah=10)
			self.fail('blah should not be accepted as a keyword argument')
		except ZenException,e:
			pass
	
	def test_choose_edge(self):
		NUM_NODES = 5
		RM_EDGES = 0 #10
		
		# build a fully-connected graph
		G = Graph()
		nodes = range(NUM_NODES)
		for i in nodes:
			G.add_node(i)
			
		for i,n in enumerate(nodes):
			for nj in nodes[i+1:]:
				G.add_edge(n,nj)
				
		# remove some edges
		edges = G.edges_()
		random.shuffle(edges)
		rm_edges = edges[:RM_EDGES]
		edges = set(edges[RM_EDGES:])
		for e in rm_edges:
			G.rm_edge_(e)
		
		# check to see how the node selection goes
		for i in range(NUM_NODES*5):
			self.assertTrue(G.edge_idx(*choose_edge(G)) in edges)
			self.assertTrue(choose_edge_(G) in edges)
	
	def test_shuffle_bad_arg(self):
		G = Graph()
		G.add_edge(1,2)

		try:
			shuffle(G,blah=10)
			self.fail('blah should not be accepted as a keyword argument')
		except ZenException,e:
			pass
	
	def test_shuffle(self):
		G = Graph()
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(3,4)
		
		G2 = shuffle(G)
		
		self.assertEqual(len(G),len(G2))
		self.assertEqual(G.size(),G2.size())
		
	# def test_bad_shuffle_keep_degree(self):
	# 	G = Graph()
	# 	G.add_edge(1,2)
	# 	G.add_edge(2,3)
	# 	G.add_edge(3,4)
	# 	try:
	# 		G2 = shuffle(G,keep_degree=True)
	# 		self.fail('Shuffling this network is impossible')
	# 	except ZenException:
	# 		pass	
	
	def test_shuffle_keep_degree(self):
		G = Graph()
		G.add_edge(1,2)
		G.add_edge(2,6)
		G.add_edge(2,7)
		G.add_edge(2,3)
		G.add_edge(3,5)
		G.add_edge(5,6)
		G.add_edge(3,4)

		G2 = shuffle(G,keep_degree=True)

		d1 = ddist(G)
		d2 = ddist(G2)

		for v1,v2 in zip(d1,d2):
			self.assertEquals(v1,v2)	
		
	def test_dg_shuffle_keep_degree(self):
		G = DiGraph()
		G.add_edge(1,2)
		G.add_edge(2,3)
		G.add_edge(3,4)
		G.add_edge(4,5)
		G.add_edge(4,6)
		G.add_edge(4,7)

		G2 = shuffle(G,keep_degree=True)
		
		d1 = ddist(G)
		d2 = ddist(G2)
		
		for v1,v2 in zip(d1,d2):
			self.assertEquals(v1,v2)
			
if __name__ == '__main__':
	unittest.main()