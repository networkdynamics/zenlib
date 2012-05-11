from zen.util.benchmark import Benchmark, main

import networkx as nx
import zen as zn
import igraph as ig

import tempfile
import os

class NonRandomGraphBenchmark(Benchmark):
	
	def __init__(self):
		Benchmark.__init__(self,'Loading Edgelist Data')
		
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
				print >>fh, 'N%d N%d' % (i,j)
				
		fh.close()
		
	def teardown(self):
		
		# delete the temporary edgelist file
		# It's taken care of for us
		pass
		
	def bm_networkx(self):
		G = nx.read_edgelist(self.fname)
		
	def bm_zen(self):
		G = zn.edgelist.read(self.fname)
		
	def bm_igraph(self):
		G = ig.read(self.fname,format='ncol',weights=False)
		
if __name__ == '__main__':
	main()