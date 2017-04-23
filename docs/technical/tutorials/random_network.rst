Building a random network
=========================

This tutorial introduces how to create an undirected graph, populate it with nodes and edges, and inspect some of its intrinsic properties.

Getting started
---------------

After successfully installing, Zen, create the file ``tutorial1.py`` somewhere. This is the file we'll use in this tutorial.

The first step in working with Zen is to import the library::

	import zen
	
Creating an empty undirected network
------------------------------------

Once Zen has been imported, we can create an undirected network::

	G = zen.Graph()
	
This graph is empty upon construction.  We can see this by printing the number of nodes and edges::

	print 'Number of nodes:', len(G)
	print 'Number of edges:', G.size()
	
Graph Structure
---------------

*Nodes* are objects in a graph that represent things. *Edges* are objects in a graph that represent relationships between nodes, between things.

As will soon be shown, a node or edge can be uniquely identified in two ways:

	* Upon construction a node or edge is assigned an index, an integer which will not change for the lifetime of the graph. This index is a very high-performance way of referring to the object.
	* When constructing a node, a node *object* can be specified. If specified, this node object should be a Python object which no other node in the graph has been associated with. If no node object is specified, the node's object is None.
	* Because duplicate edges are not permitted in Zen graphs, an edge can also be uniquely identified by the node objects of its endpoints. Of course, if either of the nodes do not have an associated node object, then this identifier is invalid and cannot be used.

Most often, node objects are the most convenient way to work with a graph. The topic of node and edge indices is left for a later tutorial.

Adding nodes
------------

As the goal of this tutorial is to create a random graph, we require a set of nodes to randomly connect. We will add 100 nodes to the graph. Their node objects will be set to the integers from 0 to 99.::

	for i in range(100):
  		G.add_node(i)

If we print out the number of nodes in the graph at this point (using ``len(G)``) we will see that there are now 100 nodes in the network.

Adding edges
------------

We will now add 100 random edges to the network. This will be done by first selecting, at random, two previously created nodes. If the nodes are not already connected, we will add an edge between them.::

	import random
	i = 0
	while i < 100:
		x = random.randint(0,99)
		y = random.randint(0,99)
		if not G.has_edge(x,y):
			G.add_edge(x,y)
			i += 1
			
As shown, the function ``G.add_edge(x,y)`` will add an edge between nodes ``x`` and ``y``. As a convenience to the user, if either of these nodes are not in the graph, they will be added by the call to ``G.add_edge``.  So, if the following call is added::

	G.add_edge(1500,1600)
	
two new nodes will be added to the network (nodes ``1500`` and ``1600``). As a result, the number of nodes in the graph will now be 102.

Creating networks using generator functions
-------------------------------------------

It is also possible to create random graphs using generator functions that are available in the Zen library.

For example, to create an `Erdos-Renyi graph <http://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model>`_, we call the following::

	G = zen.generating.erdos_renyi(10,0.1)
	
See the :ref:`api-top` documentation for a complete list of the generators provided in Zen.