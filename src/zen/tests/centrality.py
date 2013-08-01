import unittest
from zen import *
import random
import time
import gc
import sys

use_networkx = False
try:
	import networkx
	use_networkx = True
except:
	pass

def timer(fxn,args):
	gc.disable()
	t1 = time.time()
	R = fxn(*args)
	t2 = time.time()
	gc.enable()
	return R, (t2-t1)

class TestEigenvectorCentrality(unittest.TestCase):
	pass
	
	# def test_random_undir_weights(self):
	# 	Gnx = networkx.Graph()
	# 	G = Graph()
	# 	
	# 	nodes = range(4)
	# 	random.shuffle(nodes)
	# 	while len(nodes) > 0:
	# 		n1 = nodes.pop()
	# 		n2 = nodes.pop()
	# 		w = random.randint(1,5)
	# 		eidx = G.add_edge(n1,n2)
	# 		G.set_weight_(eidx,w)
	# 		Gnx.add_edge(n1,n2,weight=w)
	# 		
	# 	nodes = range(4)
	# 	random.shuffle(nodes)
	# 	while len(nodes) > 0:
	# 		n1 = nodes.pop()
	# 		n2 = nodes.pop()
	# 		
	# 		if not G.has_edge(n1,n2):
	# 			w = random.randint(1,5)
	# 			eidx = G.add_edge(n1,n2)
	# 			G.set_weight_(eidx,w)
	# 			Gnx.add_edge(n1,n2,weight=w)
	# 	
	# 	Rnx,tnx = timer(networkx.eigenvector_centrality,[Gnx,100000,0.1]) #networkx.betweenness_centrality(Gnx,True,True)
	# 	R,t = timer(centrality.eigenvector_,[G,100000,0.1])
	# 	
	# 	# print 'NX:',tnx,'ZN:',t
	# 	# 
	# 	# for n in G.nodes_iter():
	# 	# 	#self.assertAlmostEqual(Rnx[n],R[G.node_idx(n)])
	# 	# 	print Rnx[n],R[G.node_idx(n)],'; ',
	# 
	# def test_random_dir_weights(self):
	# 	Gnx = networkx.DiGraph()
	# 	G = DiGraph()
	# 
	# 	nodes = range(4)
	# 	random.shuffle(nodes)
	# 	while len(nodes) > 0:
	# 		n1 = nodes.pop()
	# 		n2 = nodes.pop()
	# 		w = random.randint(1,5)
	# 		eidx = G.add_edge(n1,n2)
	# 		G.set_weight_(eidx,w)
	# 		Gnx.add_edge(n1,n2,weight=w)
	# 
	# 	nodes = range(4)
	# 	random.shuffle(nodes)
	# 	while len(nodes) > 0:
	# 		n1 = nodes.pop()
	# 		n2 = nodes.pop()
	# 
	# 		if not G.has_edge(n1,n2):
	# 			w = random.randint(1,5)
	# 			eidx = G.add_edge(n1,n2)
	# 			G.set_weight_(eidx,w)
	# 			Gnx.add_edge(n1,n2,weight=w)
	# 
	# 	Rnx,tnx = timer(networkx.eigenvector_centrality,[Gnx,100000,0.1]) #networkx.betweenness_centrality(Gnx,True,True)
	# 	R,t = timer(centrality.eigenvector_,[G,100000,0.1])
	# 
	# 	# print 'NX:',tnx,'ZN:',t
	# 	# 
	# 	# for n in G.nodes_iter():
	# 	# 	#self.assertAlmostEqual(Rnx[n],R[G.node_idx(n)])
	# 	# 	print Rnx[n],R[G.node_idx(n)],'; ',
					
class TestBetweennessCentrality(unittest.TestCase):

	def test_random_dir_weights(self):
		
		if not use_networkx:
			sys.stderr.write('Skipping TestBetweenessCentrality.test_random_dir_weights due to missing networkx library\n')
			return
		
		Gnx = networkx.DiGraph()
		G = DiGraph()
		
		nodes = range(500)
		random.shuffle(nodes)
		while len(nodes) > 0:
			n1 = nodes.pop()
			n2 = nodes.pop()
			w = random.randint(1,5)
			eidx = G.add_edge(n1,n2)
			G.set_weight_(eidx,w)
			Gnx.add_edge(n1,n2,weight=w)
			
		nodes = range(500)
		random.shuffle(nodes)
		while len(nodes) > 0:
			n1 = nodes.pop()
			n2 = nodes.pop()
			
			if not G.has_edge(n1,n2):
				w = random.randint(1,5)
				eidx = G.add_edge(n1,n2)
				G.set_weight_(eidx,w)
				Gnx.add_edge(n1,n2,weight=w)
		
		R = betweenness_centrality_(G,True,True)
		Rnx = networkx.betweenness_centrality(Gnx,normalized=True,weight='weight') #networkx.betweenness_centrality(Gnx,True,True)
		
		#print 'NX:',tnx,'ZN:',t
		
		for n in G.nodes_iter():
			self.assertAlmostEqual(Rnx[n],R[G.node_idx(n)])

	def test_random_dir_noweights(self):
		
		if not use_networkx:
			sys.stderr.write('Skipping TestBetweenessCentrality.test_random_dir_noweights due to missing networkx library\n')
			return
		
		Gnx = networkx.DiGraph()
		G = DiGraph()

		nodes = range(500)
		random.shuffle(nodes)
		while len(nodes) > 0:
			n1 = nodes.pop()
			n2 = nodes.pop()
			w = random.randint(1,5)
			eidx = G.add_edge(n1,n2)
			G.set_weight_(eidx,w)
			Gnx.add_edge(n1,n2,weight=w)

		nodes = range(500)
		random.shuffle(nodes)
		while len(nodes) > 0:
			n1 = nodes.pop()
			n2 = nodes.pop()

			if not G.has_edge(n1,n2):
				w = random.randint(1,5)
				eidx = G.add_edge(n1,n2)
				G.set_weight_(eidx,w)
				Gnx.add_edge(n1,n2,weight=w)

		R = betweenness_centrality_(G,True,False)
		Rnx = networkx.betweenness_centrality(Gnx,normalized=True,weight=None) #networkx.betweenness_centrality(Gnx,True,True)

		#print 'NX:',tnx,'ZN:',t

		for n in G.nodes_iter():
			self.assertAlmostEqual(Rnx[n],R[G.node_idx(n)])

	def test_random_udir_weights(self):
		
		if not use_networkx:
			sys.stderr.write('Skipping TestBetweenessCentrality.test_random_udir_weights due to missing networkx library\n')
			return
		
		Gnx = networkx.Graph()
		G = Graph()
		
		nodes = range(500)
		random.shuffle(nodes)
		while len(nodes) > 0:
			n1 = nodes.pop()
			n2 = nodes.pop()
			w = random.randint(1,5)
			eidx = G.add_edge(n1,n2)
			G.set_weight_(eidx,w)
			Gnx.add_edge(n1,n2,weight=w)
			
		nodes = range(500)
		random.shuffle(nodes)
		while len(nodes) > 0:
			n1 = nodes.pop()
			n2 = nodes.pop()
			
			if not G.has_edge(n1,n2):
				w = random.randint(1,5)
				eidx = G.add_edge(n1,n2)
				G.set_weight_(eidx,w)
				Gnx.add_edge(n1,n2,weight=w)
		
		R = betweenness_centrality_(G,True,True)
		Rnx = networkx.betweenness_centrality(Gnx,normalized=True,weight='weight') #networkx.betweenness_centrality(Gnx,True,True)
		
		#print 'NX:',tnx,'ZN:',t
		
		for n in G.nodes_iter():
			self.assertAlmostEqual(Rnx[n],R[G.node_idx(n)])

	def test_random_udir_noweights(self):
		
		if not use_networkx:
			sys.stderr.write('Skipping TestBetweenessCentrality.test_random_udir_noweights due to missing networkx library\n')
			return
		
		Gnx = networkx.Graph()
		G = Graph()

		nodes = range(500)
		random.shuffle(nodes)
		while len(nodes) > 0:
			n1 = nodes.pop()
			n2 = nodes.pop()
			w = random.randint(1,5)
			eidx = G.add_edge(n1,n2)
			G.set_weight_(eidx,w)
			Gnx.add_edge(n1,n2,weight=w)

		nodes = range(500)
		random.shuffle(nodes)
		while len(nodes) > 0:
			n1 = nodes.pop()
			n2 = nodes.pop()

			if not G.has_edge(n1,n2):
				w = random.randint(1,5)
				eidx = G.add_edge(n1,n2)
				G.set_weight_(eidx,w)
				Gnx.add_edge(n1,n2,weight=w)

		R = betweenness_centrality_(G,True,False)
		Rnx = networkx.betweenness_centrality(Gnx,normalized=True,weight=None) #networkx.betweenness_centrality(Gnx,True,True)

		#print 'NX:',tnx,'ZN:',t

		for n in G.nodes_iter():
			self.assertAlmostEqual(Rnx[n],R[G.node_idx(n)])

	def test_brandes_betweenness_ud(self):
		G = Graph()
		G.set_weight_(G.add_edge(0,1),3)
		G.set_weight_(G.add_edge(0,2),2)
		G.set_weight_(G.add_edge(0,3),6)
		G.set_weight_(G.add_edge(0,4),4)
		G.set_weight_(G.add_edge(1,3),5)
		G.set_weight_(G.add_edge(1,5),5)
		G.set_weight_(G.add_edge(2,4),1)
		G.set_weight_(G.add_edge(3,4),2)
		G.set_weight_(G.add_edge(3,5),1)
		G.set_weight_(G.add_edge(4,5),4)
		self.G=G
		self.exact_weighted={G.node_idx(0): 4.0, G.node_idx(1): 0.0, G.node_idx(2): 8.0, G.node_idx(3): 6.0, G.node_idx(4): 8.0, G.node_idx(5): 0.0}
		
		b = betweenness_centrality_(self.G,weighted=True,
										  normalized=False)
		for n in sorted(self.G.nodes_()):
			self.assertEqual(b[n],self.exact_weighted[n])
	
if __name__ == '__main__':
	unittest.main()