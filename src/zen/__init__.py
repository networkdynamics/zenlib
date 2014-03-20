"""
Zen is designed to be a Zero-Effort Network library - meaning that whether you just want to prototype code or need to 
implement a high-speed routine, Zen makes it possible for you to do it easily.  In order to acheive this, Zen has made
use of several specific architectural features which enable (1) ease in learning and using the library and (2) the ability
for the library to offer maximally fast routines.

Most fundamentally, every node and edge in a Zen graph has three properties:

	- an index: this is a unique integer which is assigned to the entity.  In the underlying implementation,
		this index provides a very fast lookup to information about the object, which is important when
		writing code where execution time is important.  Because this index is a direct lookup, this value
		may not be perserved across instances of the graph.
	- an object: this is a higher-level identifier for the entity.  Nodes can be given arbitrary objects, so
		long as they are hashable.  In ordinary directed and undirected graphs, the identifier for an edge is
		its endpoints (ordered endpoints, in the case of a directed graph).  The object for a node and edge
		must be both hashtable and immutable - at least with regard to the features that determine its 
		hashcode and comparison to other objects.
	- data: this is an object (often a dictionary) that contains attributes of the node or edge.
	
When operating on nodes and edges in a graph, they can be referred to using either their index or their object.
Except where it wouldn't make sense, every method that accepts nodes or edges as input has two versions - one 
that takes an object and one that takes an index.  Methods that use an index will **always** be faster than one 
that takes an object.  In order to facilitatevreadability and easy learning of method and function names, everywhere 
in Zen we use the following conventions:

 	- If a specific function accepts a node or edge, then the version that accepts objects will be called
		"<fxn>(...)".  The one which takes the index will be called "<fxn>_(...)".
		
	- Any function that is called "<fxn>(...)" and returns an edge or node will return an object. Any 
		function that is called "<fxn>_(...)" and returns an edge or node will return an index.
"""

# import graphs
from graph import *
from digraph import *
from bipartite import *
from hypergraph import *

# basic degree properties
from degree import *

# randomization routines
from randomize import *

# import generation routines
import generating

# import built-in data
import data

# import utilities
from exceptions import *
from constants import *

# import io routines
from io import *

# import algorithms
from algorithms import *

# import control functionality
import control

# import visual stuff
import layout
from drawing import *
from view import *

# import interoperability
from nx import *
