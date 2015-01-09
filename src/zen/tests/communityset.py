import unittest

from zen import Graph, DiGraph
from zen.algorithms.community.communityset import CommunitySet, Community

class CommunitySetTestCase(unittest.TestCase):

	def test_digraph(self):
		G = DiGraph()

		CommunitySet(G,None,0)

class CommunityTestCase(unittest.TestCase):
	
	def test_digraph(self):
		G = DiGraph()

		Community(0,G,set())

