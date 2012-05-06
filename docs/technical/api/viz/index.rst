Visualizing network data
========================

.. toctree::
	:hidden:
	
	Layout routines <layout.rst>
	The View class <view.rst>
	Drawing routines <drawing.rst>

Visualizing a network can be thought of a being composed of three separate, but related components

	* :doc:`Layout routines <layout>` assign positions to nodes and edges.
	* the :doc:`View class <view>` augments layout information with other graphical attributes of the graph
	* :doc:`Drawing routines <drawing>` render view classes (layout + graphics)
	
To be concrete, layout and drawing capabilities in Zen are connected through the :py:class:`zen.View` class, which encapsulates
both positional as well as other graphical features of a network's representation.  Layout functions accept
a network and return a ``View`` object.  This object can then be updated with specific other desired features
of the network (e.g., node shapes and edge colors).  The view object is then passed to a drawing function
which renders the network.