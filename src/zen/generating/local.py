import random

from zen.graph import Graph
from zen.digraph import DiGraph
from zen.exceptions import ZenException

__all__ = ['local_attachment']

def local_attachment(n,m,r,**kwargs):
	"""
	Generate a random graph using the local attachment model.
	
	**Args**:
	
		* ``n`` (int): the number of nodes to add to the graph
		* ``m`` (int): the number of edges a new node will add to the graph
		* ``r`` (int): the number of the ``m`` edges that will be attached to random nodes.  The remaining ``m-r`` edges will be attached
		  locally.  The value of ``r`` must be greater than or equal to 1.
	
	**KwArgs**:
		* ``directed [=False]`` (boolean): whether to build the graph directed.  If ``True``, then the ``m`` edges created
		  by a node upon its creation are instantiated as out-edges.  All others are in-edges to that node.
		* ``seed [=-1]`` (int): a seed for the random number generator
	
	**Returns**:
		:py:class:`zen.Graph` or :py:class:`zen.DiGraph`. The graph generated.  If ``directed = True``, then a :py:class:`DiGraph` will be returned.
	
	.. note::
		Source: M. O. Jackson and B. W. Rogers "Meeting Strangers and Friends of Friends: How Random are Social Networks?", The American Economic Review, 95(3), pp 890-915, 2007.
	"""
	seed = kwargs.pop('seed',None)
	
	if 'directed' in kwargs:
		raise ZenException, 'local_attachment does not support the directed argument - networks are always directed.'
	
	if type(n) != int:
		raise ZenException, 'n must be an integer'
	
	if type(m) != int:
		raise ZenException, 'm must be an integer'
	
	if type(r) != int:
		raise ZenException, 'r must be an integer'
	
	if r < 1:
		raise ZenException, 'r must be >= 1.  Got value %d' % r
	
	if len(kwargs) > 0:
		raise ZenException, 'Unknown arguments: %s' % ', '.join(kwargs.keys())
		
	if seed is None:
		seed = -1
		
	G = DiGraph()
	
	random.seed(seed)
	
	#####
	# seed the model with nodes and edges
	for i in range(m+1):
		G.add_node(i)
		
	for i in range(m+1):
		for j in range(m+1):
			if i == j:
				continue
			else:
				G.add_edge(i,j)
		
	#####
	# Grow the network
	for i in range(m+1,n):
		G.add_node(i)
		
		rnd_parents = [random.randint(0,i-1) for j in range(r)]
		rnd_neighbors = set()
		for p in rnd_parents:
			rnd_neighbors.update(G.out_neighbors(p))
		
		# remove all parent nodes from the neighbors
		rnd_neighbors.difference_update(rnd_parents)
		
		# select some random neighbors
		rnd_neighbors = list(rnd_neighbors)
		random.shuffle(rnd_neighbors)
		
		for n in rnd_parents:
			G.add_node(i,n)
		for n in rnd_neighbors[m-r:]:
			G.add_node(i,n)
			
	# done
	return G
