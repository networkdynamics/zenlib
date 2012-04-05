import unittest

# import all unit tests

# base
from tests.graph import *
from tests.digraph import *
from tests.hypergraph import *

# generating functions
from tests.generating_er import *
from tests.generating_ba import *

# io
from tests.edgelist import *
from tests.memlist import *
from tests.rdot import *
from tests.scn import *
from tests.bel import *

# analysis & properties
from tests.properties import *
from tests.clustering import *
from tests.centrality import *

# shortest paths
from tests.shortest_path import *
from tests.floyd_warshall import *
from tests.bellman_ford import *
from tests.dijkstra import *
from tests.unweighted_sssp import *

# matching algorithms
from tests.max_matching import *
 
# graph generation
from tests.randomize import *
 
# spanning algorithms
from tests.spanning import *
 
# utilities
from tests.fiboheap import *
 
# modularity
from tests.modularity import *

# label propagation for community detection
#from tests.label_propagation import *

#layout
from tests.layout import *

# drawing
from tests.drawing import *

# built-in data
from tests.data import *

if __name__ == '__main__':
	unittest.main()
