from zen.util.benchmark import Benchmark, main

import networkx as nx
import zen as zn
import igraph as ig

import tempfile
import os

class NonRandomGraphBenchmark(Benchmark):
	
	def __init__(self):
		Benchmark.__init__(self,'Random Graph Creation')
		
		self.NUM_NODES = 1000

	def setup(self):
		
		# make a temporary edgelist file
		fd,fname = tempfile.mkstemp()
		os.close(fd)
		
		self.fname = fname
		
		# the graph is fully connected
		fh = open(self.fname,'w')
		for i in range(self.NUM_NODES):
			for j in range(i+1,self.NUM_NODES):
				print >>fh, '%d %d' % (i,j)
				
		fh.close()
		
	def teardown(self):
		
		# delete the temporary edgelist file
		# It's taken care of for us
		pass
		
	# def bm_igraph(self):
	# 	G = ig.Graph()
	# 	
	# 	# add nodes
	# 	node_lookup = {}
	# 	G.add_vertices(self.NUM_NODES)
	# 	for i in range(self.NUM_NODES):
	# 		node_lookup[i] = i
	# 	
	# 	# add edges
	# 	num_nodes = self.NUM_NODES
	# 	i = 0
	# 	ni = 0
	# 	nj = 1
	# 	edges = []
	# 	while i < self.NUM_EDGES:
	# 		edges.append( (ni,nj) )
	# 		i += 1
	# 		
	# 		nj += 1
	# 		if nj >= num_nodes:
	# 			ni += 1
	# 			nj = ni + 1
	# 	 	
	# 	G.add_edges(edges)
		
	def bm_networkx(self):
		G = nx.read_edgelist(self.fname)
		
	def bm_zen(self):
		G = zn.edgelist.read(self.fname)
		
if __name__ == '__main__':
	main()