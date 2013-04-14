
import logging

import xmlrpclib

logger = logging.getLogger(__name__)

class UbigraphRenderer:
	
	def __init__(self,graph,url,auto_clear=True):
		self.graph = graph
		self.node_map = {}
		self.edge_map = {}
		self.graph.add_listener(self)
		
		logger.debug('connecting to ubigraph server: %s' % url)
		self.server = xmlrpclib.Server(url)
		self.server_graph = self.server.ubigraph
		if auto_clear:
			logger.info('clearing server graph')
			self.server_graph.clear()
		
		# setup node and edge defaults
		self.server_graph.set_vertex_style_attribute(0, 'color', '#0000bb')
		self.server_graph.set_vertex_style_attribute(0, 'shape', 'sphere')

		self.server_graph.set_edge_style_attribute(0, 'color', '#ffffff')
		self.server_graph.set_edge_style_attribute(0, 'width', '1.0')
		
		# build up the graph as it currently exists
		for nidx,nobj,data in self.graph.nodes_iter_(obj=True,data=True):
			self.node_added(nidx,nobj,data)
			
		for eidx,data,weight in self.graph.edges_iter_(data=True,weight=True):
			uidx,vidx = self.graph.endpoints_(eidx)
			self.edge_added(eidx,uidx,vidx,data,weight)
		
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
			
if __name__ == '__main__':
	import zen
	import time
	
	logging.basicConfig(level=logging.DEBUG)
	
	G = zen.Graph()
	ur = UbigraphRenderer(G,'http://localhost:20738/RPC2')
	
	G.add_edge(1,2)
	time.sleep(1)
	G.add_edge(2,3)
	time.sleep(1)
	G.add_edge(3,4)
	time.sleep(1)
	G.add_edge(1,4)
	# time.sleep(1)
	# G.add_edge(5,1)
	# time.sleep(1)
	# G.add_edge(5,2)
	# time.sleep(1)
	# G.add_edge(5,3)
	# time.sleep(1)
	# G.add_edge(5,4)
	
	