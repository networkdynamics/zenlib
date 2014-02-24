import unittest

import zen

class UndirectedDuplicationDivergenceIKYTestCase(unittest.TestCase):
	
	def test_bad_argument(self):
		try:
			zen.generating.duplication_divergence_iky(10,0.5,blah=10)
			self.fail('blah should not be accepted as a keyword argument')
		except zen.ZenException,e:
			pass
			
	def test_seed(self):
		G1 = zen.generating.duplication_divergence_iky(10,0.5,seed=10)
		G2 = zen.generating.duplication_divergence_iky(10,0.5,seed=10)
		
		for e in G1.edges_iter():
			if not G2.has_edge(*e):
				self.fail('Graphs generated using same seed are different.')
				
		for e in G2.edges_iter():
			if not G1.has_edge(*e):
				self.fail('Graphs generated using same seed are different.')
	
	def test_basic(self):
		G = zen.generating.duplication_divergence_iky(10,0.5)
		
		self.assertEquals(len(G),10)
				
class DirectedDuplicationDivergenceIKYTestCase(unittest.TestCase):

	def test_bad_argument(self):
		try:
			zen.generating.duplication_divergence_iky(10,0.5,blah=10,directed=True)
			self.fail('blah should not be accepted as a keyword argument')
		except zen.ZenException,e:
			pass

	def test_directed(self):
		G1 = zen.generating.duplication_divergence_iky(10,0.5,seed=10,directed=True)
		
		self.assertIsInstance(G1,zen.DiGraph)
		
	def test_seed(self):
		G1 = zen.generating.duplication_divergence_iky(10,0.5,seed=10,directed=True)
		G2 = zen.generating.duplication_divergence_iky(10,0.5,seed=10,directed=True)

		for e in G1.edges_iter():
			if not G2.has_edge(*e):
				self.fail('Graphs generated using same seed are different.')

		for e in G2.edges_iter():
			if not G1.has_edge(*e):
				self.fail('Graphs generated using same seed are different.')

	def test_basic(self):
		G = zen.generating.duplication_divergence_iky(10,0.5,directed=True)

		self.assertEquals(len(G),10)