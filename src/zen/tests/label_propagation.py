import unittest
from zen import Graph
from zen.algorithms.community.label_propagation import *

class TestLabelPropagation(unittest.TestCase):
    def test_one_community(self):
        # Create a graph that is essentially one community
        graph = Graph()
        graph.add_edge(1,2)
        graph.add_edge(1,3)
        graph.add_edge(1,4)
        graph.add_edge(2,3)
        graph.add_edge(2,4)
        graph.add_edge(3,4)

        dictionary = label_propagation(graph)
        for node, label in dictionary.iteritems():
            self.assertEqual(label, 0)

    def test_two_communities(self):
        # Create a graph that should give two separate communities
        graph = Graph()
        # The first community
        graph.add_edge(1,2)
        graph.add_edge(1,3)
        graph.add_edge(2,3)
        graph.add_edge(2,5)
        graph.add_edge(2,4)
        graph.add_edge(3,5)
        graph.add_edge(4,5)
        
        # The other community
        graph.add_edge(6,7)
        graph.add_edge(7,8)
        graph.add_edge(6,8)
        graph.add_edge(7,9)
        graph.add_edge(6,9)
        graph.add_edge(8,9)

        # Link between the communities
        graph.add_edge(5,6)

        # should give us 0 and 1
        # First community = 0, second community = 1
        dictionary = label_propagation(graph)
        for node, label in dictionary.iteritems():
            if node >= 1 and node <= 5:
                self.assertEqual(label, 0)
            else:
                self.assertEqual(label, 1)
