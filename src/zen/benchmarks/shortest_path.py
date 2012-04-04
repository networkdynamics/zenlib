from zen.util.benchmark import Benchmark
import networkx as nx
import zen as zn
import igraph as ig
import random

class AllPairsBenchmark(Benchmark):

	def __init__(self):
		Benchmark.__init__(self,'All Pairs Comparison')
		
		self.NUM_NODES = 200
		self.NUM_SOURCES = 20
		self.P = 0.05
		
	def setup(self):
		# build an ER network
		self.ER_G = zn.Graph()	

		for i in range(self.NUM_NODES):
			self.ER_G.add_node(i)

		# add edges
		for i in range(self.NUM_NODES):
			for j in range(i+1,self.NUM_NODES):
				if random.random() < self.P:
					self.ER_G.add_edge_(i,j)

	def bm_floyd_warshall(self):
		zn.floyd_warshall_path_length_(self.ER_G)
		
	def bm_apsp(self):
		zn.all_pairs_shortest_path_length_(self.ER_G)
		
	def bm_apsp_dijkstra(self):
		zn.all_pairs_dijkstra_path_length_(self.ER_G)

class UUERSSSPBenchmark(Benchmark):
	
	def __init__(self):
		Benchmark.__init__(self,'Unweighted SSSP')
		
		self.NUM_NODES = 200
		self.NUM_SOURCES = 20
		self.P = 0.05
		
	def setup(self):
		# build an ER network
		self.ER_G = zn.Graph()	
		
		for i in range(self.NUM_NODES):
			self.ER_G.add_node(i)
			
		# add edges
		for i in range(self.NUM_NODES):
			for j in range(i+1,self.NUM_NODES):
				if random.random() < self.P:
					self.ER_G.add_edge_(i,j)
					
		# generate some endpoints
		self.sources = random.sample(range(self.NUM_NODES),self.NUM_SOURCES)
					
	def bm_zen(self):
		for i in self.sources:
			zn.single_source_shortest_path(self.ER_G, i)

	def bm_zenopt(self):
		for i in self.sources:
			zn.single_source_shortest_path_(self.ER_G, i)
		
	def setup_networkx(self):
		self.nx_ER_G = nx.Graph()
		
		for i in self.ER_G.nodes_iter():
			self.nx_ER_G.add_node(i)
			
		for i,j in self.ER_G.edges_iter():
			self.nx_ER_G.add_edge(i,j)
		
	def bm_networkx(self):
		for i in self.sources:
			nx.single_source_shortest_path(self.nx_ER_G, i)
			
	def setup_igraph(self):
		self.ig_ER_G = ig.Graph()

		self.ig_ER_G.add_vertices(len(self.ER_G))
		self.ig_ER_G.add_edges(self.ER_G.edges())

	def bm_igraph(self):
		for i in self.sources:
			self.ig_ER_G.get_shortest_paths(i)
			
class UERDijkstraBenchmark(Benchmark):

	def __init__(self):
		Benchmark.__init__(self,'Dijkstra')

		self.NUM_NODES = 200
		self.NUM_SOURCES = 20
		self.P = 0.05

	def setup(self):
		# build an ER network
		self.ER_G = zn.Graph()	

		for i in range(self.NUM_NODES):
			self.ER_G.add_node(i)

		# add edges
		for i in range(self.NUM_NODES):
			for j in range(i+1,self.NUM_NODES):
				if random.random() < self.P:
					self.ER_G.add_edge_(i,j)

		# generate some endpoints
		self.sources = random.sample(range(self.NUM_NODES),self.NUM_SOURCES)

	def bm_zen(self):
		for i in self.sources:
			zn.dijkstra_path(self.ER_G, i)

	def bm_zenopt(self):
		for i in self.sources:
			zn.dijkstra_path_(self.ER_G, i)

	def setup_networkx(self):
		self.nx_ER_G = nx.Graph()

		for i in self.ER_G.nodes_iter():
			self.nx_ER_G.add_node(i)

		for i,j in self.ER_G.edges_iter():
			self.nx_ER_G.add_edge(i,j)

	def bm_networkx(self):
		for i in self.sources:
			nx.single_source_dijkstra(self.nx_ER_G, i)
			
	def setup_igraph(self):
		self.ig_ER_G = ig.Graph()

		self.ig_ER_G.add_vertices(len(self.ER_G))
		self.ig_ER_G.add_edges(self.ER_G.edges())

	def bm_igraph(self):
		for i in self.sources:
			self.ig_ER_G.get_shortest_paths(i)
			
class UERFloydWarshallBenchmark(Benchmark):

	def __init__(self):
		Benchmark.__init__(self,'Floyd-Warshall')

		self.NUM_NODES = 200
		self.NUM_SOURCES = 20
		self.P = 0.05

	def setup(self):
		# build an ER network
		self.ER_G = zn.Graph()	

		for i in range(self.NUM_NODES):
			self.ER_G.add_node(i)

		# add edges
		for i in range(self.NUM_NODES):
			for j in range(i+1,self.NUM_NODES):
				if random.random() < self.P:
					self.ER_G.add_edge_(i,j)

	def bm_zen(self):
		zn.floyd_warshall_path_length(self.ER_G)

	def bm_zenopt(self):
		zn.floyd_warshall_path_length_(self.ER_G)

	def setup_networkx(self):
		self.nx_ER_G = nx.Graph()

		for i in self.ER_G.nodes_iter():
			self.nx_ER_G.add_node(i)

		for i,j in self.ER_G.edges_iter():
			self.nx_ER_G.add_edge(i,j)

	def bm_networkx(self):
		nx.floyd_warshall(self.nx_ER_G)

	def setup_igraph(self):
		self.ig_ER_G = ig.Graph()

		self.ig_ER_G.add_vertices(len(self.ER_G))
		self.ig_ER_G.add_edges(self.ER_G.edges())

	#def bm_igraph(self):
		#pass
		#self.ig_ER_G.shortest_paths()
		
class UUERAPSPBenchmark(Benchmark):

	def __init__(self):
		Benchmark.__init__(self,'Unweighted APSP')

		self.NUM_NODES = 200
		self.NUM_SOURCES = 20
		self.P = 0.05

	def setup(self):
		# build an ER network
		self.ER_G = zn.Graph()	

		for i in range(self.NUM_NODES):
			self.ER_G.add_node(i)

		# add edges
		for i in range(self.NUM_NODES):
			for j in range(i+1,self.NUM_NODES):
				if random.random() < self.P:
					self.ER_G.add_edge_(i,j)

	def bm_zen(self):
		zn.all_pairs_shortest_path_length(self.ER_G)

	def bm_zenopt(self):
		zn.all_pairs_shortest_path_length_(self.ER_G)

	def setup_networkx(self):
		self.nx_ER_G = nx.Graph()

		for i in self.ER_G.nodes_iter():
			self.nx_ER_G.add_node(i)

		for i,j in self.ER_G.edges_iter():
			self.nx_ER_G.add_edge(i,j)

	def bm_networkx(self):
		nx.all_pairs_shortest_path_length(self.nx_ER_G)

	def setup_igraph(self):
		self.ig_ER_G = ig.Graph()

		self.ig_ER_G.add_vertices(len(self.ER_G))
		self.ig_ER_G.add_edges(self.ER_G.edges())

	def bm_igraph(self):
		self.ig_ER_G.shortest_paths()