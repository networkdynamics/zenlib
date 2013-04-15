"""
The ``zen.drawing.ubigraph`` module provides support for rendering Zen graphs in the `Ubigraph visualization environment <http://ubietylab.net/ubigraph/>`_.  The renderer will update the visualization in real time as changes are made to the underlying graph.  Furthermore, edges and nodes can be visually highlighted.

The functionality of this module falls into two areas: rendering the topology of the graph and highlighting nodes and edges.  All this functionality is
available through the :py:class:`zen.UbigraphRenderer` class.

Rendering a graph
=================

In order to render a graph, first construct the `UbigraphRenderer` and connect it to an Ubigraph server.  

A simple use case involving a connection to a local Ubigraph server would look something like::

	G = Graph()
	ur = UbigraphRenderer('http://localhost:20738/RPC2')
	ur.graph = G
	ur.default_node_color = '#00ff00' # all nodes will be green
	
	G.add_edge(1,2)
	G.add_edge(2,3)
	
In this example, the graph is empty at first.  Because the renderer registers as a graph event listener, the Ubigraph view
will be updated as nodes and edges are added.

Note that it is possible to change the way that nodes and edges will be rendered by default.  Currently the following attributes
are supported:

	* ``default_node_color``
	* ``default_node_shape``
	* ``default_edge_color``
	* ``default_edge_width``
	
All these attributes assume values dictated by the `Ubigraph API <http://ubietylab.net/ubigraph/content/Docs/index.html>`_.

Node/Edge Highlighting
======================

Nodes and edges can be highlighted using the methods :py:meth:`zen.UbigraphRenderer.highlight_nodes`/:py:meth:`zen.UbigraphRenderer.highlight_nodes_` and :py:meth:`zen.UbigraphRenderer.highlight_edges`/:py:meth:`zen.UbigraphRenderer.highlight_edges_`.  As always, the underscore allows use of either node/edge indices (with the underscore) or node/edge objects (without the underscore).

The UbigraphRenderer class
==========================

.. autoclass:: zen.UbigraphRenderer()

"""
import logging

import xmlrpclib

logger = logging.getLogger(__name__)

class UbigraphRenderer(object):
	"""
	The UbigraphRenderer is constructed with a URL to the Ubigraph server it will connect to.  Following this, the graph can be set using the ``.graph`` attribute.
	"""
	def __init__(self,url):
		logger.debug('connecting to ubigraph server: %s' % url)
		self.server = xmlrpclib.Server(url)
		self.server_graph = self.server.ubigraph
		
		self.default_node_color = '#0000bb'
		self.default_node_shape = 'sphere'
		self.default_edge_color = '#ffffff'
		self.default_edge_width = '1.0'
		
		self.highlighted_node_color = '#bb0000'
		self.highlighted_node_shape = 'sphere'
		self.highlighted_edge_color = '#ffff00'
		self.highlighted_edge_width = '6.0'

	def __graph(self,graph=None):
		if graph is None:
			return self._graph
		else:
			self.server_graph.clear()
			
			# reapply defaults
			self.default_node_color = self._default_node_color
			self.default_node_shape = self._default_node_shape
			self.default_edge_color = self._default_edge_color
			self.default_edge_width = self._default_edge_width
			
			self.highlighted_node_color = self._hlight_node_color
			self.highlighted_node_shape = self._hlight_node_shape
			self.highlighted_edge_color = self._hlight_edge_color
			self.highlighted_edge_width = self._hlight_edge_width
			
			# zero out highlighted anything
			self._highlighted_edges = set()
			self._highlighted_nodes = set()
			
			# initialize graph stuff
			self._graph = graph
			self.node_map = {}
			self.edge_map = {}
			self._graph.add_listener(self)
	
			# build up the graph as it currently exists
			for nidx,nobj,data in self._graph.nodes_iter_(obj=True,data=True):
				self.node_added(nidx,nobj,data)
		
			for eidx,data,weight in self._graph.edges_iter_(data=True,weight=True):
				uidx,vidx = self._graph.endpoints_(eidx)
				self.edge_added(eidx,uidx,vidx,data,weight)
	
	graph = property( __graph, __graph)
	
	def __inner_default_node_color(self,color=None):
		"""
		If a color is given, the default node color is changed.  Otherwise, the default color is returned.
		"""
		if color is not None:
			self.server_graph.set_vertex_style_attribute(0, 'color', color)
			self._default_node_color = color
		else:
			return self._default_node_color

	def __inner_default_node_shape(self,shape=None):
		"""
		If a shape is given, the default node shape is changed.  Otherwise, the default shape is returned.
		"""
		logger.debug('entering inner default node shape with %s' % shape)
		if shape is not None:
			self.server_graph.set_vertex_style_attribute(0, 'shape', shape)
			self._default_node_shape = shape
		else:
			return self._default_node_shape
			
	def __inner_default_edge_color(self,color=None):
		"""
		If a shape is given, the default edge color is changed.  Otherwise, the default color is returned.
		"""
		if color is not None:
			self.server_graph.set_edge_style_attribute(0, 'color', color)
			self._default_edge_color = color
		else:
			return self._default_edge_color
			
	def __inner_default_edge_width(self,width=None):
		"""
		If a width (string) is given, the default edge width is changed.  Otherwise, the default width is returned.
		"""
		if width is not None:
			self.server_graph.set_edge_style_attribute(0, 'width', width)
			self._default_edge_width = width
		else:
			return self._default_edge_width
	
	default_node_color = property(__inner_default_node_color, __inner_default_node_color)
	default_node_shape = property(__inner_default_node_shape, __inner_default_node_shape)
		
	default_edge_color = property(__inner_default_edge_color, __inner_default_edge_color)
	default_edge_width = property(__inner_default_edge_width, __inner_default_edge_width)
	
	def __inner_hlight_node_color(self,color=None):
		"""
		If a color is given, the highlighted node color is changed.  Otherwise, the highlighted color is returned.
		"""
		if color is not None:
			self._hlight_node_color = color
		else:
			return self._hlight_node_color

	def __inner_hlight_node_shape(self,shape=None):
		"""
		If a shape is given, the hlight node shape is changed.  Otherwise, the hlight shape is returned.
		"""
		logger.debug('entering inner hlight node shape with %s' % shape)
		if shape is not None:
			self._hlight_node_shape = shape
		else:
			return self._hlight_node_shape
			
	def __inner_hlight_edge_color(self,color=None):
		"""
		If a shape is given, the hlight edge color is changed.  Otherwise, the hlight color is returned.
		"""
		if color is not None:
			self._hlight_edge_color = color
		else:
			return self._hlight_edge_color
			
	def __inner_hlight_edge_width(self,width=None):
		"""
		If a width (string) is given, the hlight edge width is changed.  Otherwise, the hlight width is returned.
		"""
		if width is not None:
			self._hlight_edge_width = width
		else:
			return self._hlight_edge_width
	
	highlighted_node_color = property(__inner_hlight_node_color, __inner_hlight_node_color)
	highlighted_node_shape = property(__inner_hlight_node_shape, __inner_hlight_node_shape)
		
	highlighted_edge_color = property(__inner_hlight_edge_color, __inner_hlight_edge_color)
	highlighted_edge_width = property(__inner_hlight_edge_width, __inner_hlight_edge_width)
																	
	def node_added(self,nidx,nobj,data):
		# skip nodes that have already been seen
		if nidx in self.node_map:
			logger.warn('node %d cannot be added. A mapping already exists.' % nidx)
			return
			
		logger.debug('registering node %d with the server' % nidx)
		self.node_map[nidx] = self.server_graph.new_vertex()
		self.server_graph.set_vertex
		
		return
		
	def node_removed(self,nidx,nobj):
		if nidx in self.node_map:
			logger.debug('removing node %d from the server.' % nidx)
			self.server_graph.remove_vertex(self.node_map[nidx])
			del self.node_map[nidx]
		else:
			logger.warn('node %d cannot be removed. No mapping exists.' % nidx)
		
	def edge_added(self,eidx,uidx,vidx,data,weight):
		# skip nodes that have already been seen
		if eidx in self.edge_map:
			logger.warn('edge %d cannot be added. A mapping already exists.' % eidx)
			return
			
		logger.debug('registering edge %d with the server' % eidx)
		self.edge_map[eidx] = self.server_graph.new_edge(self.node_map[uidx],self.node_map[vidx])
		
		return
		
	def edge_removed(self,eidx,uidx,vidx):
		if eidx in self.edge_map:
			logger.debug('removing edge %d from the server.' % eidx)
			self.server_graph.remove_edge(self.edge_map[eidx])
			del self.edge_map[eidx]
		else:
			logger.warn('edge %d cannot be removed. No mapping exists.' % eidx)
	
	def highlight_edges_(self,edges):
		for eidx in edges:
			if eidx not in self._highlighted_edges:
				self.server_graph.set_edge_attribute(self.edge_map[eidx], 'color', self._hlight_edge_color)
				self.server_graph.set_edge_attribute(self.edge_map[eidx], 'width', self._hlight_edge_width)
				self._highlighted_edges.add(eidx)
				
		return
		
	def highlight_nodes_(self,nodes):
		for nidx in nodes:
			if nidx not in self._highlighted_nodes:
				self.server_graph.set_vertex_attribute(self.node_map[nidx], 'color', self._hlight_node_color)
				self.server_graph.set_vertex_attribute(self.node_map[nidx], 'shape', self._hlight_node_shape)
				self._highlighted_nodes.add(nidx)
				
		return
		
	def highlight_edges(self,edges):
		self.highlight_edges_(map(lambda x: self._graph.edge_idx(*x),edges))
		
	def highlight_nodes(self,nodes):
		self.highlight_nodes_(map(lambda x: self._graph.node_idx(x),nodes))
		
		
if __name__ == '__main__':
	import zen
	import time
	
	logging.basicConfig(level=logging.DEBUG)
	
	G = zen.Graph()
	ur = UbigraphRenderer('http://localhost:20738/RPC2')
	ur.default_node_shape = 'sphere'
	ur.default_node_color = '#1100dd'
	ur.graph = G
	
	e1 = G.add_edge(1,2)
	time.sleep(1)
	e2 = G.add_edge(2,3)
	time.sleep(1)
	e3 = G.add_edge(3,4)
	time.sleep(1)
	e4 = G.add_edge(1,4)

	ur.highlight_edges([(1,2),(2,3)])
	ur.highlight_nodes([1])
	
	