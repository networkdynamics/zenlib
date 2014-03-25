The undirected graph
====================

.. autoclass:: zen.Graph()


Graph construction
------------------

.. automethod:: zen.Graph.__init__([node_capacity=100,edge_capacity=100,edge_list_capacity=5])

.. automethod:: zen.Graph.copy()

.. automethod:: zen.Graph.from_adj_matrix(M,...)

Basic graph properties
----------------------

.. method:: zen.Graph.is_directed()

.. automethod:: zen.Graph.__len__()

.. automethod:: zen.Graph.size()

.. automethod:: zen.Graph.is_compact()

.. automethod:: zen.Graph.validate([verbose=False])

.. automethod:: zen.Graph.matrix()

Modifying the graph
-------------------

.. automethod:: zen.Graph.compact()

Nodes
~~~~~

.. automethod:: zen.Graph.add_nodes(num_nodes[,node_obj_fxn=None])

.. automethod:: zen.Graph.add_node([nobj=None,data=None])

.. automethod:: zen.Graph.add_node_x(node_idx, edge_list_capacity, nobj, data)

.. automethod:: zen.Graph.rm_node(nobj)

.. automethod:: zen.Graph.rm_node_(nidx)

.. automethod:: zen.Graph.set_node_object(curr_node_obj,new_node_obj)

.. automethod:: zen.Graph.set_node_object_(node_idx,new_node_obj)

.. automethod:: zen.Graph.set_node_data(nobj,data)

.. automethod:: zen.Graph.set_node_data_(nidx,data)

Edges
~~~~~

.. automethod:: zen.Graph.add_edge(u,v[, data=None, weight=1])

.. automethod:: zen.Graph.add_edge_(u,v[, data=None, weight=1])

.. automethod:: zen.Graph.add_edge_x(eidx,u,v,data,weight)

.. automethod:: zen.Graph.rm_edge(u,v)

.. automethod:: zen.Graph.rm_edge_(eidx)

.. automethod:: zen.Graph.set_edge_data(u,v,data)

.. automethod:: zen.Graph.set_edge_data_(eidx,data)

.. automethod:: zen.Graph.set_weight(u,v,weight)

.. automethod:: zen.Graph.set_weight_(eidx,weight)

Accessing nodes and edges
-------------------------

Nodes
~~~~~

.. automethod:: zen.Graph.__contains__(nobj)

.. automethod:: zen.Graph.node_idx(nobj)

.. automethod:: zen.Graph.node_object(nidx)

.. automethod:: zen.Graph.__getitem__(nobj)

.. automethod:: zen.Graph.node_data(nobj)

.. automethod:: zen.Graph.node_data_(nidx)

.. automethod:: zen.Graph.degree(nobj)

.. automethod:: zen.Graph.degree_(nidx)

.. automethod:: zen.Graph.nodes([data=False])

.. automethod:: zen.Graph.nodes_([obj=False,data=False])

.. automethod:: zen.Graph.neighbors(nobj[,data=False])

.. automethod:: zen.Graph.neighbors_(nidx[,obj=False,data=False])

Edges
~~~~~

.. automethod:: zen.Graph.has_edge(u,v)

.. automethod:: zen.Graph.has_edge_(u,v)

.. automethod:: zen.Graph.endpoints(eidx)

.. automethod:: zen.Graph.endpoints_(eidx)

.. automethod:: zen.Graph.endpoint(eidx,u)

.. automethod:: zen.Graph.endpoint_(eidx,u)

.. automethod:: zen.Graph.edge_idx(u,v)

.. automethod:: zen.Graph.edge_idx_(u,v)

.. automethod:: zen.Graph.edge_data(u,v)

.. automethod:: zen.Graph.edge_data_(eidx)

.. automethod:: zen.Graph.weight(u,v)

.. automethod:: zen.Graph.weight_(eidx)

.. automethod:: zen.Graph.edges([nobj=None,data=False,weight=False])

.. automethod:: zen.Graph.edges_([nidx=-1,data=False,weight=False])

Graph Event Listeners
---------------------

Instances of a graph can notify one or more listeners of changes to it.  Listeners should support the following methods:

	* ``node_added(nidx,nobj,data)``
	* ``node_removed(nidx,nobj)``
	* ``edge_added(eidx,uidx,vidx,data,weight)``
	* ``edge_removed(eidx,uidx,vidx)``
	
Other event notifications are possible (changes to data, etc...).  These will be supported in future versions.

It is noteworthy that adding listeners imposes a serious speed limitation on graph building functions.  If no listeners
are present in the graph, then node/edge addition/removal proceed as fast as possible.  Notifying listeners requires 
these functions to follow non-optimal code paths.

.. automethod:: zen.Graph.add_listener(listener)

.. automethod:: zen.Graph.rm_listener(listener)

Iterating over the graph
------------------------

Nodes
~~~~~

.. automethod:: zen.Graph.nodes_iter([data=False])

.. automethod:: zen.Graph.nodes_iter_([obj=False,data=False])

.. automethod:: zen.Graph.neighbors_iter(nobj[,data=False])

.. automethod:: zen.Graph.neighbors_iter_(nidx[,obj=False,data=False])

.. automethod:: zen.Graph.grp_neighbors_iter(nbunch[,data=False])

.. automethod:: zen.Graph.grp_neighbors_iter_(nbunch[,obj=False,data=False])

Edges
~~~~~

.. automethod:: zen.Graph.edges_iter([nobj=None,data=False,weight=False])

.. automethod:: zen.Graph.edges_iter_([nidx=-1,data=False,weight=False])

.. automethod:: zen.Graph.grp_edges_iter(nbunch[,data=False,weight=False])

.. automethod:: zen.Graph.grp_edges_iter_(nbunch[,data=False,weight=False])
