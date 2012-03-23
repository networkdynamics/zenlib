"""
A class that encapsulates visual properties of a network.
"""
from exceptions import *

__all__ = ['View','CIRCLE']

NAN = float('nan')

CIRCLE = 'circle'

SHAPES = [CIRCLE]
SHAPE_DIMS = {
				CIRCLE:1, # radius
			}

class View:
	"""
	The base class for views of a network in zen.
	"""
	def __init__(self,G,pos=None):
		self.G = G
		
		self._pos = pos
		
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
		return self.G
	
	def set_shape_(self,nidx,shape_info):
		if shape_info is not None:
			self.__check_shape(shape_info)
			self._nshape[nidx] = shape_info
			
			max_radius = self.__shape_radius(shape_info[0],shape_info[1])
			if max_radius > self._max_shape_radius:
				self._max_shape_radius = max_radius
		else:
			del self._nshape[nidx]
			
	def shape_(self,nidx,use_default=True):
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
		# this will raise an exception if the shape/dim is invalid
		self.__check_shape(shape_info)
		
		self._default_nshape = shape_info[0]
		self._default_nshape_dim = shape_info[1]
		
		max_radius = self.__shape_radius(shape_info[0],shape_info[1])
		if max_radius > self._max_shape_radius:
			self._max_shape_radius = max_radius
		
	def get_default_shape(self):
		return self._default_nshape, self._default_nshape_dim
	
	def has_pos(self):
		return self._pos != None
	
	def set_pos_array(self,pos):
		self._pos = pos
	
	def pos_array(self):
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
		if nidx not in self._ncolors:
			if use_default:
				return self._default_ncolor
			else:
				return None
		else:
			return self._ncolors[nidx]
		
	def set_node_color_(self,nidx,color):
		if color is not None:
			self._ncolors[nidx] = color
		elif nidx in self._ncolors:
			del self._ncolors[nidx]
		
	def edge_color_(self,eidx,use_default=True):
		if eidx not in self._ecolors:
			if use_default:
				return self._default_ecolor
			else:
				return None
		else:
			return self._ecolors[eidx]

	def set_edge_color_(self,eidx,color):
		if color is not None:
			self._ecolors[eidx] = color
		elif eidx in self._ecolors:
			del self._ecolors[eidx]

	def edge_width_(self,eidx,use_default=True):
		if eidx not in self._ewidths:
			if use_default:
				return self._default_ewidth
			else:
				return None
		else:
			return self._ewidths[eidx]

	def set_edge_width_(self,eidx,width):
		if width is not None:
			self._ewidths[eidx] = width
		elif eidx in self._ewidths:
			del self._ewidths[eidx]
					
	def default_node_border(self):
		return self._default_nborder

	def set_default_node_border(self,border_info):
		self._default_nborder = border_info
		
	def default_node_color(self):
		return self._default_ncolor
		
	def set_default_node_color(self,color):
		self._default_ncolor = color

	def default_edge_color(self):
		return self._default_ecolor
		
	def set_default_edge_color(self,color):
		self._default_ecolor = color
		
	def default_edge_width(self):
		return self._default_ewidth

	def set_default_edge_width(self,width):
		self._default_ewidth = width