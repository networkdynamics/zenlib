The bipartite graph
====================

.. automodule:: zen.bipartite

In this section, only those features of the :py:class:`zen.BipartiteGraph` that differ from its parent class, :py:class:`zen.Graph` are documented.
For a comprehensive set of features (methods and properties), see the :py:class:`zen.Graph` reference documentation.

Graph construction
------------------

.. automethod:: zen.BipartiteGraph.__init__([node_capacity=100,edge_capacity=100,edge_list_capacity=5])

Modifying the graph
-------------------

Nodes
~~~~~

.. automethod:: zen.BipartiteGraph.add_node_by_class(as_u[,nobj=None,data=None])

.. automethod:: zen.BipartiteGraph.add_u_node([nobj=None,data=None])

.. automethod:: zen.BipartiteGraph.add_v_node([nobj=None,data=None])

.. automethod:: zen.BipartiteGraph.add_node([nobj=None,data=None])

Edges
~~~~~

.. automethod:: zen.BipartiteGraph.add_edge(u,v[,data=None,weight=1])

.. automethod:: zen.BipartiteGraph.add_edge_(u,v[,data=None,weight=1])

Accessing bipartite node sets
-----------------------------

.. automethod:: zen.BipartiteGraph.U()

.. automethod:: zen.BipartiteGraph.U_()

.. automethod:: zen.BipartiteGraph.V()

.. automethod:: zen.BipartiteGraph.V_()

.. automethod:: zen.BipartiteGraph.is_in_U(nobj)

.. automethod:: zen.BipartiteGraph.is_in_U_(nidx)

.. automethod:: zen.BipartiteGraph.is_in_V(nobj)

.. automethod:: zen.BipartiteGraph.is_in_V_(nidx)
