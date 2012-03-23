from zen.util.benchmark import Benchmark

import networkx as nx
import zen as zn
import igraph as ig
import random, math

class MinimumSpanningTreeBenchmark(Benchmark):
	
	def __init__(self):
		Benchmark.__init__(self,'Minimum Spanning Tree')
		self.NUM_NODES = 2000
		self.P = 0.05
		self.weights = []
		
	def setup(self):
		# build a network
		self.ER_G = zn.Graph()

		for i in range(self.NUM_NODES):
			self.ER_G.add_node(i)

		# add edges
		for i in range(self.NUM_NODES):
			for j in range(i+1,self.NUM_NODES):
				r = random.random()
				w = random.random()
				if r < self.P:
					w = math.ceil(w*10.0)
					self.ER_G.add_edge_(i,j, weight=w)
					self.weights.append(w)

	def bm_zen(self):
		zn.spanning.minimum_spanning_tree(self.ER_G)

	def setup_networkx(self):
		self.nx_ER_G = nx.Graph()

		for i in self.ER_G.nodes_iter():
			self.nx_ER_G.add_node(i)

		for i,j, w in self.ER_G.edges_iter(weight=True):
			self.nx_ER_G.add_edge(i,j, weight=w)

	def bm_networkx(self):
		nx.minimum_spanning_tree(self.nx_ER_G)
	
	def setup_igraph(self):
		self.ig_ER_G = ig.Graph()
		self.ig_ER_G.add_vertices(len(self.ER_G))
		self.ig_ER_G.add_edges(self.ER_G.edges())
		# Edges are enumerated twice, so while building the weight vector we make sure that the opposite-direction weight wasn't added
		wweights = {}
		for i,j in self.ig_ER_G.get_edgelist():
			if i in wweights and j not in wweights[i]:
				wweights[i][j] = True 
				self.weights.append(self.ER_G.weight(i, j))

	def bm_igraph(self):
		self.ig_ER_G.spanning_tree(self.weights)