import random

from zen.graph import Graph
from zen.digraph import DiGraph
from zen.exceptions import ZenException

__all__ = ['local_attachment']

def local_attachment(n, m, r, **kwargs):
	"""
	Generate a random graph using the local attachment model.
	
	**Args**:
	
		* ``n`` (int): the number of nodes to add to the graph
		* ``m`` (int): the number of edges a new node will add to the graph
		* ``r`` (int): the number of edges (of the ``m``) that a node will add to uniformly selected random nodes.
		  All others will be added to neighbors of the ``r`` selected nodes.
	
	**KwArgs**:
		* ``seed [=-1]`` (int): a seed for the random number generator
		* ``graph [=None]`` (:py:class:`zen.DiGraph`): the graph that will be populated.  If the graph is ``None``, 
		  then a new graph will be created.
	
	**Returns**:
		:py:class:`zen.DiGraph`. The graph generated.
	
	.. note::
		Source: M. O. Jackson and B. O. Rogers "Meeting strangers and friends of friends: How random are social networks?" The American Economic Review, 2007.
	"""
	seed = kwargs.pop('seed',None)
	graph = kwargs.pop('graph',None)
	
	if graph is not None and not graph.is_directed():
		raise ZenException, 'The graph provided must be directed'
	if graph is not None and len(graph) > 0:
		raise ZenException, 'The graph provided is not empty'
	
	if len(kwargs) > 0:
		raise ZenException, 'Unknown arguments: %s' % ', '.join(kwargs.keys())
	
	if type(r) != int:
		raise ZenException, 'r must be an integer'
	elif r < 1:
		raise ZenException, 'r must be 1 or larger'
		
	if seed is None:
		seed = -1
	
	if seed >= 0:
		random.seed(seed)
		
	#####
	# build the initial graph
	
	G = graph
	if G is None:
		G = DiGraph()
	
	# populate with nodes
	for i in range(m+1):
		G.add_node(i)
		
	# according to Jackson's paper, all initial nodes have m neighbors.
	for i in range(m+1):
		for j in range(m+1):
			if j != i:
				G.add_edge(j,i)
			
	######
	# Build the rest of the graph
	node_list = list(range(m+1))
	for i in range(m+1,n):
		G.add_node(i)
		
		# pick random neighbors (the parents)
		parents = random.sample(node_list,r)
		
		# pick neighbors from the parents' neighborhoods
		potentials = set()
		for n in parents:
			potentials.update(G.out_neighbors(n))
			
		potentials.difference_update(parents)
		nsize = min([m-r,len(potentials)])
		neighbors = random.sample(potentials,nsize)
		
		# connect
		for v in (parents + neighbors):
			G.add_edge(i,v)
			
		node_list.append(i)
			
	# done
	return G

if __name__ == '__main__':
	from zen.drawing import UbigraphRenderer
	G = DiGraph()
	ur = UbigraphRenderer('http://localhost:20738/RPC2',event_delay=0.5,graph=G)

	G = local_attachment(100, 6, 4, graph=G)		

