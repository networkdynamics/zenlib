import unittest
from zen import *
import os.path as path

class RDotTestCase(unittest.TestCase):
	
	def test_test1(self):
		G = rdot.read(path.join(path.dirname(__file__),'test1.rdot'))
		
		self.assertEqual(len(G),4)
		self.assertEqual(G.size(),4)
		