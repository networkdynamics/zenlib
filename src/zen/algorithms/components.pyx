"""
This module provides functions for identifying and measuring the components present in a graph.
"""

from zen.graph cimport Graph
from zen.digraph cimport DiGraph
from zen.exceptions import *

__all__ = ['components']

cpdef components(G):
	"""
	Returns a list of the connected components in graph G.  Each component is a set of the node objects in the components.
	"""
	if type(G) == Graph:
		return ug_components(<Graph> G)
	elif type(G) == DiGraph:
		raise InvalidGraphTypeException, 'DiGraph is not yet supported'
		#return dg_components(<DiGraph> G)
	else:
		raise InvalidGraphTypeException, 'Only Graph and DiGraph objects are supported'
	
cpdef ug_components(Graph G):
	all_nodes = set(G.nodes())

	components = []

	while len(all_nodes) > 0:
		seed = all_nodes.pop()
		visited = set()
		to_visit = set([seed])
		while len(to_visit) > 0:
			n = to_visit.pop()
			visited.add(n)
			nbrs = set(G.neighbors(n))
			nbrs.difference_update(visited)
			to_visit.update(nbrs)

		components.append(visited)
		all_nodes.difference_update(visited)

	return components

cpdef dg_components(DiGraph G):
	all_nodes = set(G.nodes())

	components = []

	while len(all_nodes) > 0:
		seed = all_nodes.pop()
		visited = set()
		to_visit = set([seed])
		while len(to_visit) > 0:
			n = to_visit.pop()
			visited.add(n)
			nbrs = set(G.out_neighbors(n))
			nbrs.difference_update(visited)
			to_visit.update(nbrs)

		components.append(visited)
		all_nodes.difference_update(visited)

	return components
