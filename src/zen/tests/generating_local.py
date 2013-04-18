import unittest

import zen

class LocalAttachmentTestCase(unittest.TestCase):
	
	def test_bad_argument(self):
		try:
			zen.generating.local_attachment(10,3,1,blah=10)
			self.fail('blah should not be accepted as a keyword argument')
		except zen.ZenException,e:
			pass
	
	def test_seed(self):
		G1 = zen.generating.local_attachment(10,3,1,seed=10)
		G2 = zen.generating.local_attachment(10,3,1,seed=10)
		
		for e in G1.edges_iter():
			if not G2.has_edge(*e):
				self.fail('Graphs generated using same seed are different.')
				
		for e in G2.edges_iter():
			if not G1.has_edge(*e):
				self.fail('Graphs generated using same seed are different.')
	
	def test_min_r(self):
		try:
			zen.generating.local_attachment(10,3,0)
			self.fail('r=0 should not be accepted')
		except zen.ZenException,e:
			pass
		
	def test_same_num_edges(self):
		G1 = zen.generating.local_attachment(50,10,2)
		G2 = zen.generating.local_attachment(50,10,2)

		self.assertEqual(G1.size(),G2.size())
		self.assertEqual(len(G1),len(G2))
