The directed graph
====================

.. autoclass:: zen.DiGraph()


Graph construction
------------------

.. automethod:: zen.DiGraph.__init__([node_capacity=100,edge_capacity=100, edge_list_capacity=5])

.. automethod:: zen.DiGraph.copy()

.. automethod:: zen.DiGraph.reverse()

.. automethod:: zen.DiGraph.skeleton([data_merge_fxn=NO_NONE_LIST_OF_DATA, weight_merge_fxn=AVG_OF_WEIGHTS])

.. automethod:: zen.DiGraph.from_adj_matrix(M,...)

Basic graph properties
----------------------

.. method:: zen.DiGraph.is_directed()

.. automethod:: zen.DiGraph.__len__()

.. automethod:: zen.DiGraph.size()

.. automethod:: zen.DiGraph.is_compact()

.. automethod:: zen.DiGraph.validate([verbose=False])

.. automethod:: zen.DiGraph.matrix()

Modifying the graph
-------------------

.. automethod:: zen.DiGraph.compact()

Nodes
~~~~~

.. automethod:: zen.DiGraph.add_nodes(num_nodes[,node_obj_fxn=None])

.. automethod:: zen.DiGraph.add_node([nobj=None,data=None])

.. automethod:: zen.DiGraph.add_node_x(node_idx, in_edge_list_capacity, out_edge_list_capacity, nobj, data)

.. automethod:: zen.DiGraph.rm_node(nobj)

.. automethod:: zen.DiGraph.rm_node_(nidx)

.. automethod:: zen.DiGraph.set_node_object(curr_node_obj,new_node_obj)

.. automethod:: zen.DiGraph.set_node_object_(node_idx,new_node_obj)

.. automethod:: zen.DiGraph.set_node_data(nobj,data)

.. automethod:: zen.DiGraph.set_node_data_(nidx,data)

Edges
~~~~~

.. automethod:: zen.DiGraph.add_edge(src,tgt[,data=None, weight=1])

.. automethod:: zen.DiGraph.add_edge_(src,tgt[,data=None, weight=1])

.. automethod:: zen.DiGraph.add_edge_x(eidx,src,tgt,data,weight)

.. automethod:: zen.DiGraph.rm_edge(src,tgt)

.. automethod:: zen.DiGraph.rm_edge_(eidx)

.. automethod:: zen.DiGraph.set_edge_data(src,tgt,data)

.. automethod:: zen.DiGraph.set_edge_data_(eidx,data)

.. automethod:: zen.DiGraph.set_weight(src,tgt,weight)

.. automethod:: zen.DiGraph.set_weight_(eidx,weight)

Accessing nodes and edges
-------------------------

Nodes
~~~~~

.. automethod:: zen.DiGraph.__contains__(nobj)

.. automethod:: zen.DiGraph.node_idx(nobj)

.. automethod:: zen.DiGraph.node_object(nidx)

.. automethod:: zen.DiGraph.__getitem__(nobj)

.. automethod:: zen.DiGraph.node_data(nobj)

.. automethod:: zen.DiGraph.node_data_(nidx)

.. automethod:: zen.DiGraph.degree(nobj)

.. automethod:: zen.DiGraph.degree_(nidx)

.. automethod:: zen.DiGraph.in_degree(nobj)

.. automethod:: zen.DiGraph.in_degree_(nidx)

.. automethod:: zen.DiGraph.out_degree(nobj)

.. automethod:: zen.DiGraph.out_degree_(nidx)

.. automethod:: zen.DiGraph.nodes([data=False])

.. automethod:: zen.DiGraph.nodes_([obj=False,data=False])

.. automethod:: zen.DiGraph.neighbors(nobj[,data=False])

.. automethod:: zen.DiGraph.neighbors_(nidx[,obj=False,data=False])

.. automethod:: zen.DiGraph.in_neighbors(nobj[,data=False])

.. automethod:: zen.DiGraph.in_neighbors_(nidx[,obj=False,data=False])

.. automethod:: zen.DiGraph.out_neighbors(nobj[,data=False])

.. automethod:: zen.DiGraph.out_neighbors_(nidx[,obj=False,data=False])

Edges
~~~~~

.. automethod:: zen.DiGraph.has_edge(u,v)

.. automethod:: zen.DiGraph.has_edge_(u,v)

.. automethod:: zen.DiGraph.endpoints(eidx)

.. automethod:: zen.DiGraph.endpoints_(eidx)

.. automethod:: zen.DiGraph.endpoint(eidx,u)

.. automethod:: zen.DiGraph.endpoint_(eidx,u)

.. automethod:: zen.DiGraph.src(eidx)

.. automethod:: zen.DiGraph.src_(eidx)

.. automethod:: zen.DiGraph.tgt(eidx)

.. automethod:: zen.DiGraph.tgt_(eidx)

.. automethod:: zen.DiGraph.edge_idx(u,v)

.. automethod:: zen.DiGraph.edge_idx_(u,v)

.. automethod:: zen.DiGraph.edge_data(u,v)

.. automethod:: zen.DiGraph.edge_data_(eidx)

.. automethod:: zen.DiGraph.weight(u,v)

.. automethod:: zen.DiGraph.weight_(eidx)

.. automethod:: zen.DiGraph.edges([nobj=None,data=False,weight=False])

.. automethod:: zen.DiGraph.edges_([nidx=-1,data=False,weight=False])

.. automethod:: zen.DiGraph.in_edges(nobj[,data=False,weight=False])

.. automethod:: zen.DiGraph.in_edges_(nidx[,data=False,weight=False])

.. automethod:: zen.DiGraph.out_edges(nobj[,data=False,weight=False])

.. automethod:: zen.DiGraph.out_edges_(nidx[,data=False,weight=False])

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

.. automethod:: zen.DiGraph.add_listener(listener)

.. automethod:: zen.DiGraph.rm_listener(listener)

Iterating over the graph
------------------------

Nodes
~~~~~

.. automethod:: zen.DiGraph.nodes_iter([data=False])

.. automethod:: zen.DiGraph.nodes_iter_([obj=False,data=False])

.. automethod:: zen.DiGraph.neighbors_iter(nobj[,data=False])

.. automethod:: zen.DiGraph.neighbors_iter_(nidx[,obj=False,data=False])

.. automethod:: zen.DiGraph.in_neighbors_iter(nobj[,data=False])

.. automethod:: zen.DiGraph.in_neighbors_iter_(nidx[,obj=False,data=False])

.. automethod:: zen.DiGraph.out_neighbors_iter(nobj[,data=False])

.. automethod:: zen.DiGraph.out_neighbors_iter_(nidx[,obj=False,data=False])

.. automethod:: zen.DiGraph.grp_neighbors_iter(nbunch[,data=False])

.. automethod:: zen.DiGraph.grp_neighbors_iter_(nbunch[,obj=False,data=False])

.. automethod:: zen.DiGraph.grp_in_neighbors_iter(nbunch[,data=False])

.. automethod:: zen.DiGraph.grp_in_neighbors_iter_(nbunch[,obj=False,data=False])

.. automethod:: zen.DiGraph.grp_out_neighbors_iter(nbunch[,data=False])

.. automethod:: zen.DiGraph.grp_out_neighbors_iter_(nbunch[,obj=False,data=False])

Edges
~~~~~

.. automethod:: zen.DiGraph.edges_iter([nobj=None,data=False,weight=False])

.. automethod:: zen.DiGraph.edges_iter_([nidx=-1,data=False,weight=False])

.. automethod:: zen.DiGraph.in_edges_iter(nobj[,data=False,weight=False])

.. automethod:: zen.DiGraph.in_edges_iter_(nidx[,data=False,weight=False])

.. automethod:: zen.DiGraph.out_edges_iter(nobj[,data=False,weight=False])

.. automethod:: zen.DiGraph.out_edges_iter_(nidx[,data=False,weight=False])

.. automethod:: zen.DiGraph.grp_edges_iter(nbunch[,data=False,weight=False])

.. automethod:: zen.DiGraph.grp_edges_iter_(nbunch[,data=False,weight=False])

.. automethod:: zen.DiGraph.grp_in_edges_iter(nbunch[,data=False,weight=False])

.. automethod:: zen.DiGraph.grp_in_edges_iter_(nbunch[,data=False,weight=False])

.. automethod:: zen.DiGraph.grp_out_edges_iter(nbunch[,data=False,weight=False])

.. automethod:: zen.DiGraph.grp_out_edges_iter_(nbunch[,data=False,weight=False])
