import unittest

from zen import *
from zen.algorithms.community import *

import numpy as np

#TODO Test empty community / communityset

class CommunityTestCase(unittest.TestCase):
	
	def setUp(self):
		self.NUM_NODES = 5

		self.G = Graph()
		for i in range(self.NUM_NODES):
			self.G.add_node(i)

		self.fake_community = Community(0, self.G, 
				{i for i in range(self.NUM_NODES)})

	def test_has_node(self):

		# Test presence of all inserted nodes in the community
		for i in range(self.NUM_NODES):
			self.assertTrue(self.fake_community.has_node_index(i))
			self.assertTrue(i in self.fake_community)

		# Test absence of nodes
		self.assertFalse(self.fake_community.has_node_index(self.NUM_NODES))
		self.assertFalse(self.NUM_NODES in self.fake_community)

class CommunitySetTestCase(unittest.TestCase):
	
	def setUp(self):
		self.G = Graph()
		self.NUM_NODES = 10

		for i in range(self.NUM_NODES):
			self.G.add_node(i)

		# Communities, with nodes that belong to it:
		# 0: 0,1 / 1: 2,3 / 2: 4,5 / 3: 6,7 / 4: 8,9
		fake_community_table = np.array([i / 2 for i in range(self.NUM_NODES)])
		self.cmnties = CommunitySet(self.G, fake_community_table, self.NUM_NODES / 2)

	def test_community_idx(self):

		# Test that the nodes belong to the expected community
		for i in range(self.NUM_NODES):
			self.assertEqual(i / 2, self.cmnties.community_idx(i))
			self.assertEqual(i / 2, self.cmnties.community_idx_(i))

		# Test that an exception is raised for nodes that are not in the set
		self.assertRaises(ZenException, self.cmnties.community_idx_, self.NUM_NODES)
		self.assertRaises(KeyError, self.cmnties.community_idx, self.NUM_NODES)

	def test_share_community(self):

		# Test that communities are shared as expected (by how cmnties is built)
		for i in range(0, self.NUM_NODES, 2):
			self.assertTrue(self.cmnties.share_community(i, i + 1))
			self.assertTrue(self.cmnties.share_community_(i, i + 1))

		# Test reflexivity of community sharing
		for i in range(self.NUM_NODES):
			self.assertTrue(self.cmnties.share_community(i, i))
			self.assertTrue(self.cmnties.share_community_(i, i))

		# Test symmetry of community sharing
		for i in range(self.NUM_NODES - 1):
			if(self.cmnties.share_community(i, i + 1)):
				self.assertTrue(self.cmnties.share_community(i + 1, i))
			else:
				self.assertFalse(self.cmnties.share_community(i + 1, i))

			if(self.cmnties.share_community_(i, i + 1)):
				self.assertTrue(self.cmnties.share_community_(i + 1, i))
			else:
				self.assertFalse(self.cmnties.share_community_(i + 1, i))

		#Test that no node shares a community with a node not in the graph
		for i in range(self.NUM_NODES):
			self.assertFalse(self.cmnties.share_community(i, self.NUM_NODES))
			self.assertRaises(ZenException, self.cmnties.share_community_, i, self.NUM_NODES)

	def test_community(self):
		for i in range(self.NUM_NODES):

			com_idx = self.cmnties.community_(i)
			com_obj = self.cmnties.community(i)

			# Test that the communities returned contain the objects they are
			# supposed to
			self.assertTrue(com_idx.has_node_index(i))
			self.assertTrue(i in com_obj)

			if i % 2 == 1:
				other = i - 1
			else:
				other = i + 1

			self.assertTrue(com_idx.has_node_index(other))
			self.assertTrue(other in com_idx)

			# Test that the given community indices are as expected
			self.assertEqual(com_idx.community_idx, self.cmnties.community_idx_(i))
			self.assertEqual(com_obj.community_idx, self.cmnties.community_idx(i))

			# Test that the communities contain only the expected nodes
			self.assertEqual(len(com_idx), 2)
			self.assertEqual(len(com_obj), 2)

	#TODO communities
	def test_communities(self):
		pass

	#TODO iteration
	def test_iteration(self):
		pass

	#TODO length
	def test_len(self):
		pass

class CommunityDetectionTestCase(unittest.TestCase):

	def setUp(self):
		self.algorithms = [ label_propagation, 
							label_rank, 
							spectral_modularity, 
							louvain ]
		#Algorithms that are supposed to work on graphs that were not compacted
		self.noncontiguous_algorithms = [	label_propagation, 
											label_rank, 
											louvain ]

		self.empty = Graph()

		self.disconnected = Graph()
		self.disconnected.add_edge(0, 1)
		self.disconnected.add_edge(2, 3)

		self.k3 = Graph()
		self.k3.add_edge(0, 1)
		self.k3.add_edge(1, 2)
		self.k3.add_edge(2, 0)

		self.noncontiguous = Graph()
		self.noncontiguous.add_node(0)
		self.noncontiguous.add_node(1)
		self.noncontiguous.add_node(2)
		self.noncontiguous.add_node(3)
		self.noncontiguous.add_edge(0, 2)
		self.noncontiguous.add_edge(2, 3)
		self.noncontiguous.rm_node(1)


	def __algorithm_test_empty(self, algorithm):
		# Test that no communities are detected on an empty graph
		cset = algorithm(self.empty)
		self.assertEqual(len(cset.communities()), 0)

	def __algorithm_test_disconnected(self, algorithm):
		# Test that two communities are detected on a disconnected graph with
		# two components
		cset = algorithm(self.disconnected)
		self.assertEqual(len(cset.communities()), 2)

	def __algorithm_test_k3(self, algorithm):
		# Test that one community is detected on a K3
		cset = algorithm(self.k3)
		self.assertEqual(len(cset.communities()), 1)

	def __algorithm_test_noncontiguous_nidc(self, algorithm):
		# Test that the algorithm doesn't fail if the node indices are not
		# contiguous
		try:
			algorithm(self.noncontiguous)
		except Exception as e:
			self.fail("Noncontiguous test failed with " + e.__class__.__name__ + 
						": \"" + str(e) + "\"")

	def test_sanity_algorithms(self):
		# Some simple graphs with obvious answers
		for alg in self.algorithms:

			self.__algorithm_test_empty(alg)
			self.__algorithm_test_disconnected(alg)
			self.__algorithm_test_k3(alg)

		for alg in self.noncontiguous_algorithms:
			self.__algorithm_test_noncontiguous_nidc(alg)

if __name__ == '__main__':
	unittest.main()
