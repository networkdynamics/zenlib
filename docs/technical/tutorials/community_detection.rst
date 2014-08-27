Detecting communities in networks
=================================

This tutorial gives an overview of the various community detection methods
available in Zen. Broadly speaking, a community in a network can be defined as 
*a group of nodes that are more connected to each other than to the rest of the
network*. Community detection is the process of assigning nodes to communities
so that the assignments respect the true structure of the network. A lot of
algorithms have been developed to solve this problem and Zen provides several of
them through the ``zen.algorithms.community`` package.

Available algorithms
--------------------

The algorithms provided by Zen can be broadly put in two categories: the
*label propagation* algorithms and the *modularity optimization* algorithms. The
first category of algorithms work more or less like this: each node is assigned
a label, which is propagated to neighboring nodes using certain rules, in such
a way that a node has the same labels as most of its neighbors. Nodes with the
same label form a community. The second category of algorithms attempt to
optimize a property of networks called modularity. Modularity is the fraction of
edges that fall within some given groups (i.e. estimated communities) minus the
fraction that would fall within the groups if edge placement was random.
Therefore, the larger the modularity, the more edges correctly fall in groups,
and thus the more those groups are close to representing communities of the 
network.


A list of the algorithms is available in the documentation for the
``zen.algorithms.community`` package.

Some algorithms may not be called on directed or weighted networks. Make sure
you check the documentation for each algorithm to know their limitations.

Calling the algorithms
----------------------

All the community detection algorithms provided by Zen have the same interface.
They take a Graph and return a CommunitySet, which is a special container holding
the data on the communities detected. Some algorithms may also take additional
arguments, which have default values provided.

For the first part of the tutorial, we are going to use LPA, which is the
canonical label-propagation algorithm. Under this algorithm, each node receives
the label used by the majority of its neighbors. Ties are broken randomly. The
algorithm terminates when all nodes have a label that agree with the majority of
their neighbors.

We are also going to use the "karate club" network. This network is famous
because it encodes the friendships between members of a karate club which split
in two because of management issues. It has thus become a benchmark of sorts for
community-detection algorithms because it is a real-life network with a known
community divide. This network is a built-in of Zen.

We will first make the necessary imports::

	import zen.algorithms.community as zac
	import zen.data

Then, we can simply call the LPA algorithm on the karate network::

	cset = zac.lpa(zen.data.karate_club())

This gives us a ``CommunitySet`` object, which we will learn how to use now.

Using community sets
--------------------

A community set is a container for communities detected in a network. Communities
are themselves containers of the nodes of this network. It is important to know
that communities may overlap, depending on the algorithm used (community-detection
algorithms can be characterized as 'overlapping' or 'non-overlapping' depending
on whether they allow communities to overlap or not). In any case, the interface
for community set and community objects is the same.

Using the community set we created above, we can easily see how many communities
have been detected by the algorithm::

	print 'Number of communities:', len(cset)

It is also possible to check whether two nodes from the network share a 
community or not::

	print 'Are nodes 0 and 1 in the same community?', cset.share_community_(0, 1)

Following the Zen convention, "underscore" methods (methods with a trailing
underscore in the name) take node indices as parameters, while non-underscore 
methods take node objects. Node indices/objects are the same in ``CommunitySet``
and ``Community`` objects and in the original network.

It is possible to obtain a list of the communities a node belongs to using the
``node_communities`` method::

	communities_of_0 = cset.node_communities_(0)

The elements of ``communities_of_0`` are ``Community`` objects. In the case of 
non-overlapping algorithms, the ``node_communities_`` method will return a list 
with one element.

It is also possible to iterate through the ``Community`` objects of a
``CommunitySet``. For example, this snippet creates a list of all the communities
of our set::

	all_communities = []
	for community in cset:
		all_communities.append(community)

However, the above code can be written more concisely as::

	all_communities = cset.communities()

Every community has an index, ranging from ``0`` to ``len(cset) - 1``. Iterating
over the community set, or the list returned by the ``communities`` method, will
yield community objects in their order of indices. To retrieve a particular
community using its index, use the ``community`` method::

	first_community = cset.community(0)

It is also possible to retrieve a list of the indices of the communities that
a node belongs to, which is faster than obtaining a list of community objects::

	communities_of_0 = cset.node_community_indices(0)

It's possible to access the index of any community object through the public
property ``community_idx``.

Now that we have seen how to obtain ``Community`` objects from a community set,
we will examine them more closely.

Using communities
-----------------

Community objects are containers too, but for nodes. This means that the usual
container semantics apply::

	first_community = cset.community(0)
	print 'Number of nodes in this community:', len(first_community)

	print 'Nodes in this community:'
	for node_index in first_community:
		print node_index,

Note that iterating over a community gives node indices, not node objects.

Membership tests can be done on node objects by using the ``in`` keyword::

	print 'Is node A in this community?', 'A' in first_community

If you want to test membership using indices instead, you can use the
``has_node_index`` method instead:

	print 'Is node 0 in this community?', first_community.has_node_index(0)

Some algorithms may return probabilistic results. For nodes that may belong to more
than one community, they will report the belonging percentage to each community
(e.g. given a node A belonging to communities 0 and 1, the algorithm may rank
A as belonging "only" 25% in community 0 and 75% in community 1). To access
this information using a ``Community`` object, the method ``assoc_prob`` 
(``assoc_prob_`` for node indices) may be used::

	print 'Probabilities of nodes belonging to this community'
	for node in first_community:
		print 'Node %s: %f' % (str(node), first_community.assoc_prob(node))

Note that this method will not throw an exception if the given node object/index
is not part of the community. Instead, it will return a probability of 0.

If this method is called on a community with no probability information (because
the algorithm didn't provide any), then it always returns 1 (unless, as stated
above, the given parameter does not correspond to a member of the community).
