"""
This module provides conversion services to and from NetworkX objects.
"""

from graph import Graph
from digraph import DiGraph
import networkx

__all__ = ['to_networkx','from_networkx']	

def to_networkx(G):
	"""
	This function accepts a Zen graph object and returns a networkx graph object that is a copy
	of the Zen graph.
	"""
	import networkx
	
	if type(G) == Graph:
		Gdest = networkx.Graph()
		
		# copy node objects and data
		for nobj,ndata in G.nodes_iter(data=True):
			if ndata is None:
				Gdest.add_node(nobj)
			else:
				Gdest.add_node(nobj,data=ndata)
		
		# copy edge objects and data
		for u,v,edata in G.edges_iter(data=True):
			if edata is None:
				Gdest.add_edge(u,v)
			else:
				Gdest.add_edge(u,v,data=edata)
				
		return Gdest
		
	elif type(G) == DiGraph:
		Gdest = networkx.DiGraph()
		
		# copy node objects and data
		for nobj,ndata in G.nodes_iter(data=True):
			if ndata is None:
				Gdest.add_node(nobj)
			else:
				Gdest.add_node(nobj,data=ndata)
		
		# copy edge objects and data
		for u,v,edata in G.edges_iter(data=True):
			if edata is None:
				Gdest.add_edge(u,v)
			else:
				Gdest.add_edge(u,v,data=edata)
				
		return Gdest
	else:
		raise ZenException, 'Cannot convert objects of type %s to NetworkX graph objects' % str(type(G))
	
# def to_wrapped_networkx(G):
# 	"""
# 	This function accepts a Zen graph object and returns an object which has the networkx interface.
# 	Note that this object will wrap the graph object passed in, so any changes made to the networkx
# 	object will also be reflected in the underlying graph.  The object returned maintains no state,
# 	so changes can be made to the underlying Zen graph without affecting the validity of the 
# 	wrapper.
# 	"""
# 	import networkx
# 	
# 	if type(G) == DiGraph:
# 		return DiGraphNXWrapper(G)
# 	else:
# 		raise Exception, 'Unable to convert graph object type %s' % str(type(G))
	
def from_networkx(G):
	"""
	This function converts a graph from a networkx data structure into a Zen graph.
	"""
	Gdest = None
	if type(G) == networkx.DiGraph:
		Gdest = DiGraph()
	elif type(G) == networkx.Graph:
		Gdest = Graph()
	else:
		raise Exception, 'Unable to convert graph object type %s' % str(type(G))
	
	# add nodes	
	for n,nd in G.nodes_iter(data=True):
		Gdest.add_node(n,nd)
		
	# add edges
	for x,y,ed in G.edges_iter(data=True):
		Gdest.add_edge(x,y,ed)
		
	return Gdest