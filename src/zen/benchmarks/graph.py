from zen.util.benchmark import Benchmark
from random import random

import networkx as nx
import zen as zn
import igraph as ig


class NonRandomGraphBenchmark(Benchmark):
	
	def __init__(self):
		Benchmark.__init__(self,'Random Graph Creation')
		
		self.NUM_NODES = 100000
		self.NUM_EDGES = 100000

	# This is WAY to slow due to the cost of adding edges individually
	# def bm_igraph2(self):
	# 	G = ig.Graph()
	# 
	# 	# add nodes
	# 	G.add_vertices(self.NUM_NODES)
	# 
	# 	# add edges
	# 	num_nodes = self.NUM_NODES
	# 	i = 0
	# 	ni = 0
	# 	nj = 1
	# 	edges = []
	# 	while i < self.NUM_EDGES:
	# 		G.add_edges( (ni,nj) )
	# 		i += 1
	# 
	# 		nj += 1
	# 		if nj >= num_nodes:
	# 			ni += 1
	# 			nj = ni + 1
		
	def bm_igraph(self):
		G = ig.Graph()
		
		# add nodes
		node_lookup = {}
		G.add_vertices(self.NUM_NODES)
		for i in range(self.NUM_NODES):
			node_lookup[i] = i
		
		# add edges
		num_nodes = self.NUM_NODES
		i = 0
		ni = 0
		nj = 1
		edges = []
		while i < self.NUM_EDGES:
			edges.append( (ni,nj) )
			i += 1
			
			nj += 1
			if nj >= num_nodes:
				ni += 1
				nj = ni + 1
		 	
		G.add_edges(edges)
		
	def bm_networkx(self):
		G = nx.Graph()
		
		# add nodes
		for i in range(self.NUM_NODES):
			G.add_node(i)
			
		# add edges
		num_nodes = self.NUM_NODES
		i = 0
		ni = 0
		nj = 1
		while i < self.NUM_EDGES:
			G.add_edge(ni,nj)
			i += 1
			
			nj += 1
			if nj >= num_nodes:
				ni += 1
				nj = ni + 1
		
	def bm_zen(self):
		G = zn.Graph()
		
		# add nodes
		for i in range(self.NUM_NODES):
			G.add_node(i)
			
		# add edges
		num_nodes = self.NUM_NODES
		i = 0
		ni = 0
		nj = 1
		while i < self.NUM_EDGES:
			G.add_edge(ni,nj)
			i += 1
			
			nj += 1
			if nj >= num_nodes:
				ni += 1
				nj = ni + 1
			
	def bm_zenopt(self):
		
		G = zn.Graph()
		
		# add nodes
		G.add_nodes(self.NUM_NODES)
			
		# add edges
		num_nodes = self.NUM_NODES
		i = 0
		ni = 0
		nj = 1
		while i < self.NUM_EDGES:
			G.add_edge_(ni,nj)
			i += 1
			
			nj += 1
			if nj >= num_nodes:
				ni += 1
				nj = ni + 1
			
	# def bm_zenopt(self):
	# 	G = zn.Graph()
	# 	
	# 	# add nodes
	# 	for i in range(self.NUM_NODES):
	# 		G.add_node()
	# 		
	# 	# add edges
	# 	num_nodes = self.NUM_NODES
	# 	i = 0
	# 	ni = 0
	# 	nj = 1
	# 	while i < self.NUM_EDGES:
	# 		G.add_edge_(ni,nj)
	# 		i += 1
	# 		
	# 		nj += 1
	# 		if nj >= num_nodes:
	# 			ni += 1
	# 			nj = ni + 1