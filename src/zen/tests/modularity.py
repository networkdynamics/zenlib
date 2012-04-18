import unittest
from zen import *

class TestSynonymModularity(unittest.TestCase):
    def test_small_graph_modularity(self):
		graph=Graph()
		graph.add_node('1')
		graph.add_node('2')
		graph.add_node('3')
		graph.add_node('4')
		graph.add_node('5')
		graph.add_node('6')
		graph.add_edge('1','2')
		graph.add_edge('3','2')
		graph.add_edge('1','3')
		graph.add_edge('4','5')
		graph.add_edge('4','6')
		graph.add_edge('5','6')
		graph.add_edge('2','5')
		community_assignment={0:['1','2','3'],1:['4','5','6']}
		expected_result=10.0/28 # hand calculated
		#print 'modularity: ' + str(modularity.modularity(graph, community_assignment))
		self.assertAlmostEqual(modularity(graph, community_assignment), expected_result)
		graph.add_node('7')
		graph.add_node('8')
		graph.add_node('9')
		graph.add_edge('7','8')
		graph.add_edge('8','9')
		graph.add_edge('9','7')
		graph.add_edge('8','5')
		graph.add_edge('8','2')
		expected_result=5.0/12 # hand calculated
		
		community_assignment={0:['1','2','3'],1:['4','5','6'],2:['7','8','9']}
		#print 'modularity: ' + str(modularity.modularity(graph, community_assignment))
		self.assertAlmostEqual(modularity(graph, community_assignment), expected_result)
		
		#community_assignment={0:['1','3'],1:['4','6'],2:['7','9'],3:['2','5','8']}
		#print 'modularity: ' + str(modularity.modularity(graph, community_assignment))
		
		
		
		
		
		
        
        
		
		
		
		
    '''	
    def test_words(self):
        # Create the graph first so we can use it for the tests
        # Using adjacency lists to store synonyms of a word
        synonyms = []

        synonyms.append(['free', 'complimentary', 'at liberty', 'costless', 'gratis', 'unpaid', 'liberated', 'unrestricted', 'unlimited', 'open', 'familiar', 'permitted', 'independent', 'idle'])
        synonyms.append(['complimentary', 'costless', 'flattering', 'polite', 'gratis'])
        synonyms.append(['gratis', 'free', 'costless', 'donated'])
        synonyms.append(['unpaid', 'donated', 'volunteer'])
        synonyms.append(['unrestricted', 'unlimited', 'at liberty', 'open', 'liberated'])
        synonyms.append(['unlimited', 'absolute', 'endless', 'boundless', 'countless', 'total', 'vast', 'infinite'])
        synonyms.append(['open', 'accessible', 'clear', 'spacious', 'wide', 'available', 'permitted', 'overt', 'plain', 'frank', 'undecided'])
        synonyms.append(['familiar', 'frequent', 'usual', 'simple'])
        synonyms.append(['permitted', 'allowed', 'acceptable', 'approved', 'tolerated'])
        synonyms.append(['independent', 'liberated', 'self-reliant', 'separate', 'sovereign'])
        synonyms.append(['idle', 'abandoned', 'dead', 'empty', 'untouched', 'lazy', 'pointless', 'resting', 'slothful'])
        synonyms.append(['wide', 'spacious', 'expansive', 'roomy'])
        synonyms.append(['infinite', 'absolute', 'endless', 'eternal', 'boundless', 'vast', 'wide', 'limitless', 'immense', 'total', 'untold'])
        synonyms.append(['lazy', 'slothful'])
        synonyms.append(['absolute', 'complete', 'free', 'supreme', 'unlimited', 'sovereign', 'certain', 'infallible', 'exact', 'precise'])
        synonyms.append(['immense', 'unlimited', 'vast', 'limitless', 'endless', 'boundless', 'wide'])
        synonyms.append(['limitless', 'vast', 'unlimited', 'endless', 'boundless'])
        synonyms.append(['endless', 'boundless'])

        #synonyms.append(['idle', 'abandoned', 'dead', 'empty', 'untouched', 'lazy', 'pointless', 'resting', 'slothful'])
        
        
        # Now make it into a graph
        graph = Graph()

        for adj_list in synonyms:
            # The node is always the first element
            node = adj_list[0]
            for i in range(1, len(adj_list)):
                other_node = adj_list[i]
                # Make sure there isn't already an edge
                if graph.has_edge(node, other_node):
                    #print "already exists an edge between " + node + " and " + other_node
                    pass
                else:
                    # print "adding edge between " + node + ' and ' + other_node
                    graph.add_edge(node, other_node)
        self.graph = graph
        #print graph.num_nodes
        #print graph.num_edges

        # Test a group of very similar words: 
        similar_words = dict({0:['vast', 'infinite', 'unlimited', 'endless', 'immense', 'limitless', 'boundless']})
        # print str(similar_words) + ' has modularity: ' + str(modularity.modularity(graph, similar_words))
        # Calculated by hand (+wolfram)
        expected_result = float(321) / float(4418)
        #self.assertAlmostEqual(modularity.modularity(graph, similar_words), expected_result)

        # Test a group of dissimilar words:
        dissimilar_words = {1:['open', 'lazy', 'wide', 'gratis', 'separate']}
        # print str(dissimilar_words) + ' has modularity: ' + str(modularity.modularity(graph, dissimilar_words))    
        expected_result = float(-3) / float(4418)
        self.assertAlmostEqual(modularity.modularity(graph, dissimilar_words), expected_result) 

        # Test a smaller group of similar words
        less_similar_words = {2:['limitless', 'unlimited', 'endless', 'boundless']}
        # print str(less_similar_words) + ' has modularity: ' + str(modularity.modularity(graph, less_similar_words))
        expected_result = float(867) / float(35344)
        self.assertAlmostEqual(modularity.modularity(graph, less_similar_words), expected_result)

        #community_assignment={0:[ 'frequent', 'usual', 'simple'],1:['free', 'complimentary', 'at liberty', 'costless', 'gratis', 'unpaid', 'liberated', 'unrestricted', 'unlimited', 'open', 'familiar', 'permitted', 'independent', 'idle']}
        #print 'modularity: ' + str(modularity.modularity(graph, community_assignment))
        '''
