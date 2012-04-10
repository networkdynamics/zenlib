The undirected graph
====================

.. autoclass:: zen.Graph()


Graph construction
------------------

.. autoclass:: zen.Graph()

.. automethod:: zen.Graph.copy()

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

.. automethod:: zen.Graph.set_node_object(curr_node_obj,new_node_obj)

.. automethod:: zen.Graph.set_node_object_(node_idx,new_node_obj)

.. automethod:: zen.Graph.set_node_data(nobj,data)

.. automethod:: zen.Graph.set_node_data_(nidx,data)

.. automethod:: zen.Graph.rm_node(nobj)

.. automethod:: zen.Graph.rm_node_(nidx)


Edges
~~~~~

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

.. automethod:: zen.Graph.nodes([data=False])

.. automethod:: zen.Graph.nodes_([obj=False,data=False])

.. automethod:: zen.Graph.degree(nobj)

.. automethod:: zen.Graph.degree_(nidx)

Iterating over the graph
------------------------

Nodes
~~~~~

.. automethod:: zen.Graph.nodes_iter([data=False])

.. automethod:: zen.Graph.nodes_iter_([obj=False,data=False])
