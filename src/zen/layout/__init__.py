"""
The ``zen.layout`` module provides functions for assigning positions to nodes and paths to edges according to a
selected algorithm.  Different algorithms attempt to maximize the aesthetic quality (e.g., readability) of the resulting
layout in different ways.

In order to standardize the handling of the graph visualization pipeline, all graph layout functions produce a new or 
modify an existing :py:class:`zen.View` object. Graph layout functions all return the view object produced or modified.

**Standard arguments**:

*Positional arguments.* In addition to algorithm-specific parameters, each layout function accepts one positional argument: a graph or view object.
If a graph is specified, then a new view is created.  If a view is specified, then the graph of the view (:py:meth:`zen.View.graph`)
is used and the current layout of the view is replaced by the newly generated layout.

*Keyword arguments.* All layout functions also accept the following keyword arguments:

	* ``bbox [=(0,0,100,100)]`` (:py:class:`tuple`): a tuple (left,bottom,top,right) indicating the spatial bounds that constrian the layout.
	  If set to ``None``, then positions and paths are unconstrained.  By default, the bounding box is (0,0,100,100).
	
Layout functions
~~~~~~~~~~~~~~~~

.. autofunction:: spring(GV,...)

.. autofunction:: random(GV,...)

.. autofunction:: forceatlas(GV,...)

.. autofunction:: fruchterman_reingold(GV,...)

"""

import spring_layout
import random_layout
import forceatlas_layout
import fruchtermanreingold_layout

BBOX_NAME = 'bbox'
BBOX_DEFAULT_VALUE = (0.,0.,100.,100.)

def spring(GV,**kwargs):
	"""
	Use a `spring-embedder <http://en.wikipedia.org/wiki/Force-based_algorithms_(graph_drawing)>`_ algorithm for
	deriving positions for the nodes.  Edges follow a straight path between their endpoints.
	
	**Args**:
		
		* ``GV``: the graph or view (containing a graph) whose layout will be built.
	
	**KwArgs**:
	
		* ``num_iterations [=100]`` (int): the number of times to update the positions of nodes based on the forces among them.
		* ``bbox [=(0,0,100,100)]`` (:py:class:`tuple`): the spatial box into which all nodes should be confined.
	
	**Returns**:
		:py:class:`zen.View`. A view object with the position array set to the positions determined by this layout.
	"""
	if BBOX_NAME not in kwargs:
		kwargs[BBOX_NAME] = BBOX_DEFAULT_VALUE
		
	return spring_layout.layout(GV,**kwargs)
	
def random(GV,**kwargs):
	"""
	Assign random positions to all nodes in the graph.  Edges follow a straight path between their endpoints.
	
	**Args**:
		
		* ``GV``: the graph or view (containing a graph) whose layout will be built.
		
	**KwArgs**:
	
		* ``bbox [=(0,0,100,100)]`` (:py:class:`tuple`): the spatial box into which all nodes should be confined.
	
	**Returns**:
		:py:class:`zen.View`. A view object with the position array set to the positions determined by this layout.
	"""
	if BBOX_NAME not in kwargs:
		kwargs[BBOX_NAME] = BBOX_DEFAULT_VALUE
		
	return random_layout.layout(GV,**kwargs)
	
def forceatlas(GV,**kwargs):
	"""
	Use a `force-based <http://en.wikipedia.org/wiki/Force-based_algorithms_(graph_drawing)>`_ algorithm for
	deriving positions for the nodes.  Edges follow a straight path between their endpoints.
	
	**Args**:
		
		* ``GV``: the graph or view (containing a graph) whose layout will be built.
		
	**KwArgs**:
	
		* ``bbox [=(0,0,100,100)]`` (:py:class:`tuple`): the spatial box into which all nodes should be confined.
	
	**Returns**:
		:py:class:`zen.View`. A view object with the position array set to the positions determined by this layout.
	"""
	if BBOX_NAME not in kwargs:
		kwargs[BBOX_NAME] = BBOX_DEFAULT_VALUE

	return forceatlas_layout.layout(GV,**kwargs)
	
def fruchterman_reingold(GV,**kwargs):
	"""
	Use the `Fruchterman-Reingold <http://en.wikipedia.org/wiki/Force-based_algorithms_(graph_drawing)>`_ algorithm for
	deriving positions for the nodes.  Edges follow a straight path between their endpoints.
	
	**Args**:
		
		* ``GV``: the graph or view (containing a graph) whose layout will be built.
		
	**KwArgs**:
	
		* ``num_iterations [=100]`` (int): the number of times to update the positions of nodes based on the forces among them.
		* ``bbox [=(0,0,100,100)]`` (:py:class:`tuple`): the spatial box into which all nodes should be confined.
	
	**Returns**:
		:py:class:`zen.View`. A view object with the position array set to the positions determined by this layout.
	"""
	if BBOX_NAME not in kwargs:
		kwargs[BBOX_NAME] = BBOX_DEFAULT_VALUE

	return fruchtermanreingold_layout.layout(GV,**kwargs)