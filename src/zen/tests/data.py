from zen.data import *
import unittest

class DataTestCase(unittest.TestCase):
	
	def test_florentine(self):
		G = florentine()
		
		self.assertFalse(G.is_directed())
		self.assertEqual(len(G),16)
		self.assertEqual(G.size(),19)
		
	def test_karate(self):
		G = karate_club()
		
		self.assertFalse(G.is_directed())
		self.assertEqual(len(G),34)
		self.assertEqual(G.size(),78)