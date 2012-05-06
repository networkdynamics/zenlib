"""
The ``zen.drawing.mpl`` module provides code for rendering graph views to `matplotlib <http://matplotlib.sourceforge.net/>`_ figures.

When using views with this package, the following conventions apply:

	* colors should be specified as tuples ``(R,G,B)`` where all three components are between ``0`` and ``1``
	* all dimensions (e.g., edge width and node radius) are interpreted as units in the scale of the axes of the plot used

.. autofunction:: draw(view[,fig=None,bounds=None])

.. autofunction:: draw_to_axes(view[,ax=None])
"""

import pylab as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from zen.view import *
from zen.exceptions import *

__all__ = ['draw','draw_to_axes','draw_to_figure']

ROYAL_BLUE = (65./255.,105./255.,225./255.)
STEEL_BLUE = (176./255.,196./255.,222./255.)
GREY_70 = (179./255.,179./255.,179./255.)

DEFAULT_NODE_COLOR = ROYAL_BLUE
DEFAULT_EDGE_COLOR = GREY_70

DEFAULT_EDGE_WIDTH = 1

DEFAULT_NODE_SHAPE = CIRCLE
DEFAULT_NODE_SHAPE_DIM = (2,)

def draw_to_axes(view,**kwargs):
	"""
	Draw the network view to an axes object.  
	
	If no axes are specified, then the current axes are used.  If no axes exist,
	one will be created.
	
	**Args**:
		* ``view`` (:py:class:`zen.View`): the view to draw to the figure
	
	**KwArgs**:
		* ``ax [=None]`` (matplotlib axes): the specific axes to draw the graph on.
		
	**Returns**:
		*matplotlib axes*. The axes the graph was drawn to.
	"""
	ax = kwargs.pop('ax',None)
	
	if len(kwargs) > 0:
		raise ZenException, 'Unexpected keyword arguments: %s' % ','.join(kwargs.keys())
		
	G = view.graph()
	pos = view.pos_array()
	
	if pos is None:
		raise ZenException, 'No position information is in view object'
	
	if ax == None:
		ax = pl.gca()
		
	# draw nodes
	for nidx in G.nodes_iter_():
		color = view.node_color_(nidx)
		if color == None:
			color = DEFAULT_NODE_COLOR

		bcolor = None
		bwidth = None
		binfo = view.node_border_(nidx)
		if binfo is None:
			bcolor = None
			bwidth = 0
		else:
			bcolor,bwidth = binfo
			
		x = pos[nidx,0]
		y = pos[nidx,1]
		shape,dim = view.shape_(nidx)
		
		if shape == None:
			shape = DEFAULT_NODE_SHAPE
			dim = DEFAULT_NODE_SHAPE_DIM
		
		if shape == CIRCLE:
			ax.add_patch(Circle( (x,y), radius=dim[0], facecolor=color, edgecolor=bcolor, linewidth=bwidth, zorder=2))
		else:
			raise ZenException, 'Node shape %s not supported' % shape

	# draw edges
	for eidx in G.edges_iter_():
		alpha = 1.0
		ecolor = view.edge_color_(eidx)
		if ecolor is None:
			ecolor = DEFAULT_EDGE_COLOR
		
		if len(ecolor) == 4:
			alpha = ecolor[3]
		
		ewidth = view.edge_width_(eidx)
		if ewidth is None:
			ewidth = DEFAULT_EDGE_WIDTH
			
		x,y = G.endpoints_(eidx)
		ax.add_line( Line2D( (pos[x,0],pos[y,0]), (pos[x,1],pos[y,1]), linewidth=ewidth, color=ecolor, alpha=alpha, zorder=1))
		
	return ax

def draw(view,**kwargs):
	"""
	Draw the network view to a figure.  
	
	If no figure is specified, then the current figure is used.  If no figure exists,
	one will be created.  The network will be drawn to the axes of the figure.
	
	.. note::
		This function is also defined as ``zen.drawing.mpl.draw_to_figure``.
	
	**Args**:
		* ``view`` (:py:class:`zen.View`): the view to draw to the figure
	
	**KwArgs**:
		* ``fig [=None]`` (matplotlib figure): the specific figure to draw the graph on.
		* ``bounds [=None]`` (:py:class:`tuple`): a tuple indicating the bounds of the plot. 
		  This influences the portion of the graph that will be visible and where the graph's
		  components will be positioned in the plot.  If ``None``, then bounds will be chosen
		  that fit all nodes and edges in the graph with a 10% buffer on all sides.
		
	**Returns**:
		*matplotlib figure*. The figure the graph was drawn to.
	"""
	fig = kwargs.pop('fig',None)
	bounds = kwargs.pop('bounds',None)
	
	if len(kwargs) > 0:
		raise ZenException, 'Unexpected keyword arguments: %s' % ','.join(kwargs.keys())
	
	if fig == None:
		fig = pl.gcf()
		
	ax = pl.gca()
	pl.xticks([])
	pl.yticks([])
	
	if bounds is None:
		buffer_size = float(view.max_x()-view.min_x())*0.1
		ax.set_xbound(view.min_x()-buffer_size,view.max_x()+buffer_size)
		ax.set_ybound(view.min_y()-buffer_size,view.max_y()+buffer_size)
	else:
		ax.set_xbound(bounds[0],bounds[1])
		ax.set_ybound(bounds[2],bounds[3])
		
	draw_to_axes(view,ax=ax)
	
	return fig
	
draw_to_figure = draw