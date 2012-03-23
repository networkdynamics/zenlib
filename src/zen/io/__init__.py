"""
This package provides the primary functions for reading and writing networks to and from storage (whether files, databases, or otherwise).
The only IO functionality that is not present in this package is the pickling capability that is built directly into the
Graph and DiGraph classes.

While formats and storage systems differ, we have attempted to keep the interfaces for reading and writing as uniform as
possible across the different storage systems.  Every storage system has its own module (e.g., edgelist, rdot, memlist).
Each module exposes two standard functions for reading and writing:

	- read(source,<keyword arguments>)
	- write(graph,destination,<keyword arguments>)
	
the source and destination parameters are storage system specific (e.g., for files, these will be file names).  graph is any
valid Zen graph object.  keyword arguments are used to customize the read or write operation.  While some will be storage system
specific, several are perserved across some/all storage systems.

== Read keyword arguments ==

	- node_obj_fxn [= str]: this is a function that accepts a string (the node object as a string) and returns the node object
		that should be used in the graph.  If not specified, the default functionality is to use string node objects.
		If node_obj_fxn is None, then no node objects are stored for nodes loaded.
		
	- directed [= False]: this is boolean indicating whether the topology information read should be interpreted as directed or
		undirected graph data.  The graph returned will be accordingly directed or undirected.  By default, methods assume
		data to be undirected.  Note that this parameter is not supported (or needed) by some storage systems in which 
		the directionality of the data is encoded in the source.
	
	- ignore_duplicate_edges [= False]: occasionally data sources will contain duplicate edges 
		(this is certainly true when directed data is being read as undirected data).  Setting this argument to 
		True will suppress the generation of errors related to the presence of duplicate edges in the data.  
		These will be skipped.
		
	- merge_graph [= None]: rather than creating a new network object, the data read can be merged with an existing one.  If 
		this argument is set to a valid Zen network object, then the read method will attempt to load the data into the 
		existing network.  The directionality of the existing graph will be used to decide on the interpretation of the 
		directionality of the source data read.  If the source data has an explicit directionality that disagrees with the
		network object provided, an error will be raised.  If 

== Write keyword arguments ==

	- use_node_indices [= False]: this argument indicates that the node indices should be used as the unique identifier for each
		node in the destination rather than the node objects (which is the default).  This argument is only supported by methods
		for which node indices are not required.
"""

import edgelist
import hedgelist
import rdot
import scn
import bel
import memlist