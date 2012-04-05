from zen.data import *
import unittest

class DataTestCase(unittest.TestCase):
	
	def test_florentine(self):
		
		G = florentine()
		
		self.assertEqual(len(G),16)
		self.assertEqual(G.size(),19)