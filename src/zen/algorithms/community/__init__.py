"""
The ``zen.algorithms.community`` package provides functions for detecting 
communities of nodes present within a network. Intuitively, a community is 
defined as a group of nodes which are more connected to each other than to the 
rest of the network. The following algorithms are implemented:

	- **LPA** (Label-Propagation Algorithm) detects communities by assigning 
	each node a label that the majority of its neighbors agree upon, breaking 
	ties at random. Its runtime is linear in terms of the number of edges in the
	network. However, its use of randomness means that running the algorithm on 
	the same graph several times can yield different results. It also means that
	the algorithm may sometimes fail to return a good solution (i.e. it returns 
	a single community for the entire network).

	- **LabelRank** is a stabilized (deterministic) version of LPA, which also 
	has parameters to control the propagation of the labels. Like LPA, it has a
	linear runtime in terms of edges. Depending on the given parameters, its 
	memory requirements may be higher than LPA (since it needs to maintain a 
	table of labels for each node).

	- The **spectral modularity** approach maximizes the modularity of the 
	network using properties of its adjacency matrix. It has a O(n²log n) 
	complexity, where n is the number of nodes in the network. This is worse 
	than linear complexity in terms of edges. The algorithm only supports 
	undirected, unweighted networks.

	- The **Louvain** algorithm is also a modularity-maximizing algorithm. It
	assigns each node to its own community then repeatedly moves nodes to their
	neighbors' communities in order to increase the modularity. Its complexity
	is linear in terms of nodes.
	

*Details and citations are available in the documentation for each algorithm.*

All community detection algorithms return a ``CommunitySet``, which is an object
specifically designed to contain communities (every community being contained in a 
``Community`` object). Please see the class documentation for more details.

*This package is currently under active development.*

"""

from communityset import *
from lpa import label_propagation
from label_rank import label_rank
from spectral_modularity import spectral_modularity
from louvain import louvain
