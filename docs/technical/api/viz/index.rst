Visualizing network data
========================

Visualizing a network can be thought of a being composed of two separate, but related tasks: layout and drawing.
Layout functions assign positions to nodes and edges while drawing functions render the layouts to a graphical
surface.

Layout and drawing capabilities in Zen are connected through the :py:class:`zen.View` class, which encapsulates
both positional as well as other graphical features of a network's representation.  Layout functions accept
a network and return a ``View`` object.  This object can then be updated with specific other desired features
of the network (e.g., node shapes and edge colors).  The view object is then passed to a drawing function
which renders the network.

The View class
--------------

.. automodule:: zen.view

Network layouts
---------------

In the layout phase, positions are assigned to nodes and paths through space are assigned to edges.  Typically
these positions and paths are not done at random, but rather done in order to maximize some sort of aesthetic
quality - usually readability.

Drawing functions
-----------------