"""
The ``zen.algorithms.community`` package provides functions for detecting communities of 
nodes present within a network. Intuitively, a community is defined as a group of nodes
which are more connected to each other than to the rest of the network. The following
algorithms are implemented:

	- **LPA** (Label-Propagation Algorithm) detects communities by assigning each node
	a label that the majority of its neighbors agree upon, breaking ties at random.
	Its runtime is linear in terms of the number of edges in the network. However,
	its use of randomness means that running the algorithm on the same graph several
	times can yield different results. It also means that the algorithm may sometimes
	return a trivial solution (i.e. a single community for the entire network).

	- **LabelRank** is a stabilized (deterministic) version of LPA, which also adds
	parameters to control the propagation of the labels. Like LPA, it has a linear
	runtime in terms of edges. Depending on the given parameters, its memory
	requirements may be higher than LPA.

	- The **spectral modularity** approach maximizes the modularity of the network
	using properties of its adjacency matrix. It has a O(n²log n) complexity, where n
	is the number of nodes in the network. This is worse than linear complexity in
	terms of edges. The algorithm only supports undirected, unweighted networks.

*Details and sources are available in the documentation for each algorithm.*

All community detection algorithms return a ``CommunitySet``, which is an object
specifically designed to contain communities (every community being contained in a 
``Community`` object). Please see the class documentation for more details.

*This package is currently under active development.*

"""

from label_rank import label_rank
