"""
The ``zen.View`` class encapsulates a graphical description of a graph.  This includes positional information for nodes and edges
as well as colors and shapes of individual elements.

View methods
~~~~~~~~~~~~

.. automethod:: View.__init__(G,pos=None)

.. automethod:: View.graph()

.. automethod:: View.set_shape_(nidx,shape_info)

.. automethod:: View.shape_(nidx,use_default=True)

.. automethod:: View.set_default_shape(shape_info)

.. automethod:: View.get_default_shape()

.. automethod:: View.has_pos_array()

.. automethod:: View.set_pos_array(pos_array)

.. automethod:: View.pos_array()

.. automethod:: View.node_color_(nidx,use_default=True)

.. automethod:: View.set_node_color_(nidx,color)

.. automethod:: View.edge_color_(eidx,use_default=True)

.. automethod:: View.set_edge_color_(eidx,color)

.. automethod:: View.edge_width_(eidx,use_default=True)

.. automethod:: View.set_edge_width_(eidx,width)

.. automethod:: View.default_node_color()

.. automethod:: View.set_default_node_color(color)

.. automethod:: View.default_edge_color()

.. automethod:: View.set_default_edge_color(color)

.. automethod:: View.default_edge_width()

.. automethod:: View.set_default_edge_width(color)

Shapes
~~~~~~

Shapes in the ``View`` are tuples.

.. autofunction:: circle_shape(radius)

Colors
~~~~~~

The valid range of color values/objects is determined by the specific drawing function that will be used.

"""
from exceptions import *

__all__ = ['View','CIRCLE']

NAN = float('nan')

CIRCLE = 'circle'

SHAPES = [CIRCLE]
SHAPE_DIMS = {
				CIRCLE:1, # radius
			}

def circle_shape(radius):
	"""
	Return a circle shape with the indicated radius.
	"""
	return (CIRCLE,radius)

class View:

	def __init__(self,G,pos_array=None):
		"""
		Initialize the ``View`` object with the graph, ``G``, that it will contain the graphical description for.
		
		.. note::
			This object is typically called by a call to a :py:mod:`zen.layout` functions.
		
		**Args**:
			
			* ``pos_array [=None]`` (2D ``numpy.ndarray``): the positions for each node in the graph.
		"""
		self.G = G
		
		self._pos = pos_array
		
		#####
		# Node attributes
		
		# color
		self._ncolors = {}
		self._default_ncolor = None
		
		self._nborders = {}
		self._default_nborder = None
		
		# shape
		self._nshape = {}
		self._nshape_dim = {}
		
		self._default_nshape = CIRCLE
		self._default_nshape_dim = (0.1,) # the radius of the circle
		
		self._max_shape_radius = 0.1 # the radius of the default circle
		
		#####
		# Edge attributes
		
		# color
		self._ecolors = {}
		self._default_ecolor = None
		
		self._ewidths = {}
		self._default_ewidth = None
	
	def graph(self):
		"""
		Return the graph the view is a graphical representation for.
		"""
		return self.G
	
	def set_shape_(self,nidx,shape_info):
		"""
		Set the shape of a specific node in the network.
		
		**Args**:
			* ``nidx`` (int): the node index of the node to set the shape of.
			* ``shape_info`` (:py:class:`tuple`/:py:class:`list`): the attributes of the shape. If ``shape_info`` is ``None``
			  then any shape information currently entered for this node is deleted.
		"""
		if shape_info is not None:
			self.__check_shape(shape_info)
			self._nshape[nidx] = shape_info
			
			max_radius = self.__shape_radius(shape_info[0],shape_info[1])
			if max_radius > self._max_shape_radius:
				self._max_shape_radius = max_radius
		else:
			del self._nshape[nidx]
			
	def shape_(self,nidx,use_default=True):
		"""
		Return the shape tuple for a specific node in the network.
		
		**Args**:
			* ``nidx`` (int): the node index of the node whose shape information will be retrieved.
			* ``use_default [=True]`` (boolean): if ``True`` and the node has no shape information, 
			  return the default shape information.  If ``False`` and the node has no shape information, 
			  return ``None`` (indicating that there is no shape information set for this node).
		"""
		if nidx not in self._nshape:
			if use_default:
				return self.get_default_shape()
			else:
				return None
		else:
			return self._nshape[nidx]
	
	def __check_shape(self,shape_info):
		shape = shape_info[0]
		dim = shape_info[1]
		if shape not in SHAPE_DIMS:
			raise ZenException, 'Unknown shape: %s' % shape
		if len(dim) != SHAPE_DIMS[shape]:
			raise ZenException, 'Incorrect number of dimensions for shape %s.  Expected %d, got %d.' % (shape,SHAPE_DIMS[shape],len(dim))
			
		return
	
	def __shape_radius(self,shape,shape_dims):
		if shape == CIRCLE:
			return shape_dims[0]
		else:
			raise ZenException, 'Unknown shape: %s' % shape
	
	def set_default_shape(self,shape_info):
		"""
		Set the default shape for all nodes.
		
		The shape information provided will be applied to all nodes which do not have any shape information
		specified specifically for them.
		
		**Args**:
			* ``shape_info`` (:py:class:`tuple`/:py:class:`list`): the attributes of the shape. If ``shape_info`` is ``None``
			  then any shape information currently entered for this node is deleted.
		"""
		# this will raise an exception if the shape/dim is invalid
		self.__check_shape(shape_info)
		
		self._default_nshape = shape_info[0]
		self._default_nshape_dim = shape_info[1]
		
		max_radius = self.__shape_radius(shape_info[0],shape_info[1])
		if max_radius > self._max_shape_radius:
			self._max_shape_radius = max_radius
		
	def get_default_shape(self):
		"""
		Return the default shape for nodes.
		"""
		return self._default_nshape, self._default_nshape_dim
	
	def has_pos_array(self):
		"""
		Return ``True`` if the position array for the nodes has been set.
		"""
		return self._pos != None
	
	def set_pos_array(self,pos_array):
		"""
		Set the position array for nodes in the graph.
		
		**Args**:
			* ``pos_array [=None]`` (2D ``numpy.ndarray``): the positions for each node in the graph.
		"""
		self._pos = pos_array
	
	def pos_array(self):
		"""
		Return the position array for the nodes in the network.
		"""
		return self._pos
		
	def max_x(self):
		shape_buffer = 0
		if self._max_shape_radius is not None:
			shape_buffer = self._max_shape_radius
		
		return max(self._pos[:,0]) + shape_buffer

	def max_y(self):
		shape_buffer = 0
		if self._max_shape_radius is not None:
			shape_buffer = self._max_shape_radius
			
		return max(self._pos[:,1]) + shape_buffer
		
	def min_x(self):
		shape_buffer = 0
		if self._max_shape_radius is not None:
			shape_buffer = self._max_shape_radius
			
		return min(self._pos[:,0]) - shape_buffer

	def min_y(self):
		shape_buffer = 0
		if self._max_shape_radius is not None:
			shape_buffer = self._max_shape_radius
			
		return min(self._pos[:,1]) - shape_buffer
		
	def pos_x_(self,nidx):
		if self._pos is None:
			return NAN
		else:
			return self._pos[nidx,0]
	
	def pos_y_(self,nidx):
		if self._pos is None:
			return NAN
		else:
			return self._pos[nidx,1]	
			
	def pos_(self,nidx):
		if self._pos is None:
			return NAN,NAN
		else:
			return self._pos[nidx,0], self._pos[nidx,1]

	def node_border_(self,nidx,use_default=True):
		if nidx not in self._nborders:
			if use_default:
				return self._default_nborder
			else:
				return None
		else:
			return self._nborders[nidx]
			
	def set_node_border_(self,nidx,border_info):
		if border_info is not None:
			self._nborders[nidx] = border_info
		else:
			del self._nborders[nidx]
	
	def node_color_(self,nidx,use_default=True):
		"""
		Return the color of node ``nidx``.
		
		**Args**:
			* ``nidx`` (int): the node index of the node whose color information will be retrieved.
			* ``use_default [=True]`` (boolean): if ``True`` and the node has no color, 
			  return the default color.  If ``False`` and the node has no color, 
			  return ``None`` (indicating that there is no color information set for this node).
		"""
		if nidx not in self._ncolors:
			if use_default:
				return self._default_ncolor
			else:
				return None
		else:
			return self._ncolors[nidx]
		
	def set_node_color_(self,nidx,color):
		"""
		Set the color of a node.
		
		**Args**:
			* ``nidx`` (int): the node index of the node to set the color for.
			* ``color``: the color to assign the node.
		"""
		if color is not None:
			self._ncolors[nidx] = color
		elif nidx in self._ncolors:
			del self._ncolors[nidx]
		
	def edge_color_(self,eidx,use_default=True):
		"""
		Return the color of edge ``eidx``.
		
		**Args**:
			* ``eidx`` (int): the edge index of the edge whose color information will be retrieved.
			* ``use_default [=True]`` (boolean): if ``True`` and the edge has no color, 
			  return the default color.  If ``False`` and the edge has no color, 
			  return ``None`` (indicating that there is no color information set for this edge).
		"""
		if eidx not in self._ecolors:
			if use_default:
				return self._default_ecolor
			else:
				return None
		else:
			return self._ecolors[eidx]

	def set_edge_color_(self,eidx,color):
		"""
		Set the color of an edge.
		
		**Args**:
			* ``eidx`` (int): the edge index of the edge to set the color for.
			* ``color``: the color to assign the edge.
		"""
		if color is not None:
			self._ecolors[eidx] = color
		elif eidx in self._ecolors:
			del self._ecolors[eidx]

	def edge_width_(self,eidx,use_default=True):
		"""
		Return the width of edge ``eidx``.
		
		**Args**:
			* ``eidx`` (int): the edge index of the edge whose width will be retrieved.
			* ``use_default [=True]`` (boolean): if ``True`` and the edge has no width, 
			  return the default width.  If ``False`` and the edge has no width, 
			  return ``None`` (indicating that there is no width information set for this edge).
		"""
		if eidx not in self._ewidths:
			if use_default:
				return self._default_ewidth
			else:
				return None
		else:
			return self._ewidths[eidx]

	def set_edge_width_(self,eidx,width):
		"""
		Set the width of an edge.
		
		**Args**:
			* ``eidx`` (int): the edge index of the edge to set the width for.
			* ``color``: the width to assign the edge.
		"""
		if width is not None:
			self._ewidths[eidx] = width
		elif eidx in self._ewidths:
			del self._ewidths[eidx]
					
	def default_node_border(self):
		return self._default_nborder

	def set_default_node_border(self,border_info):
		self._default_nborder = border_info
		
	def default_node_color(self):
		"""
		Return the default color that is applied to nodes.
		"""
		return self._default_ncolor
		
	def set_default_node_color(self,color):
		"""
		Set the default color that is applied to nodes.
		"""
		self._default_ncolor = color

	def default_edge_color(self):
		"""
		Return the default color that is applied to edges.
		"""
		return self._default_ecolor
		
	def set_default_edge_color(self,color):
		"""
		Set the default color that is applied to edges.
		"""
		self._default_ecolor = color
		
	def default_edge_width(self):
		"""
		Return the default width that is applied to edges.
		"""
		return self._default_ewidth

	def set_default_edge_width(self,width):
		"""
		Set the default width that is applied to nodes.
		"""
		self._default_ewidth = width