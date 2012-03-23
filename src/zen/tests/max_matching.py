from zen import *
import unittest

class HopcroftKarpTestCase(unittest.TestCase):
	
	def test_simple(self):
		G = BipartiteGraph()
		G.add_node_by_class(True,0)
		G.add_node_by_class(True,1)
		G.add_node_by_class(False,2)
		G.add_node_by_class(False,3)
		
		G.add_edge(0,2)
		G.add_edge(1,3)
		
		m = matching.maximum_matching_(G)
		
		self.assertEquals(len(m),2)

	def test_simple2(self):
		G = BipartiteGraph()
		G.add_node_by_class(True,0)
		G.add_node_by_class(True,1)
		G.add_node_by_class(True,2)
		G.add_node_by_class(True,3)

		G.add_node_by_class(False,4)
		G.add_node_by_class(False,5)
		G.add_node_by_class(False,6)
		G.add_node_by_class(False,7)

		e1 = G.add_edge(0,4)
		e2 = G.add_edge(0,5)
		e3 = G.add_edge(1,5)
		e4 = G.add_edge(1,6)
		e5 = G.add_edge(2,6)
		e6 = G.add_edge(3,7)

		m = matching.hopcroft_karp_(G)

		self.assertEquals(len(m),4)
		self.assertEquals(set(m),set([e1,e3,e5,e6]))
		
def get_driver_nodes(G):
	m = matching.maximum_matching(G)
	
	mstar = set()
	for u,v in m:
		mstar.add(v)
	
	if len(mstar) == len(G):
		# pick a random node - any one will do
		ND = G.nodes().pop()
	else:
		# get all the unmatched nodes
		ND = set(G.nodes()).difference(mstar)
		
	return ND
					
class DirectedMaximumMatchingTextCase(unittest.TestCase):
	
	def test_simple1(self):
		G = DiGraph()
		
		e1 = G.add_edge(1,2)
		e2 = G.add_edge(1,3)
		e3 = G.add_edge(3,3)
		
		m = matching.maximum_matching_(G)
		
		self.assertEquals(len(m),2)
		self.assertEquals(set(m),set([e1,e3]))
		
	def test_controllability(self):
		G = DiGraph()
		
		e1 = G.add_edge(1,2)
		e2 = G.add_edge(1,3)
		e3 = G.add_edge(3,3)
		e4 = G.add_edge(4,5)
		e5 = G.add_edge(4,6)
		e6 = G.add_edge(5,6)
		
		ND = get_driver_nodes(G)
		
		self.assertEquals(set(ND),set([1,4]))
		
if __name__ == '__main__':
	unittest.main()