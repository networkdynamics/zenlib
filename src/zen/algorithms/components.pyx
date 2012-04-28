"""
The ``zen.algorithms.components`` module provides functions related to detecting and identifying 
connected groups of nodes within graphs. All functions in this module are available at the root-level of a zen import.

In undirected graphs, interest is generally focused on identifying the 
`connected component <http://en.wikipedia.org/wiki/Connected_component_(graph_theory)>`_.  This is the
function that is available at present.

.. autofunction:: zen.algorithms.components.components(G)

.. Add this in once the functions are available.
.. In directed graphs, there are multiple types of connected components:

..	* Standard `connected components <http://en.wikipedia.org/wiki/Connected_component_(graph_theory)>`_
..	* `Strongly connected components <http://en.wikipedia.org/wiki/Strongly_connected_component>`_
	
"""

from zen.graph cimport Graph
from zen.digraph cimport DiGraph
from zen.exceptions import *

__all__ = ['components']

cpdef components(G):
	"""
	Identify all the components in the graph ``G``.
	
	**Returns**:
		:py:class:`list`, ``C``. Returns a list of the connected components in graph G.  
		Each component is a :py:class:`set` of the node objects in a specific component.
		
	**Raises**:
		:py:exc:`zen.ZenException`: if ``G`` is a :py:class:`zen.DiGraph`, since computing
		components in directed graphs is not supported yet.
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
