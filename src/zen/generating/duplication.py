import random
from sys import maxint

from zen.graph import Graph
from zen.digraph import DiGraph
from zen.exceptions import ZenException
from zen.randomize import choose_node

def duplication_divergence_iky(n, s, **kwargs):
	"""
	Generate a random graph using the duplication-divergence model proposed by Ispolatov, Krapivski, and Yuryev (2008).
	
	**Args**:
	
		* ``n`` (int): the target size of the network.
		* ``s`` (float): the probability that a link connected to a newly duplicated node will exist
	
	**KwArgs**:
		* ``directed [=False]`` (boolean): whether to build the graph directed.  If ``True``, then the ``m`` edges created
		  by a node upon its creation are instantiated as out-edges.  All others are in-edges to that node.
		* ``seed [=-1]`` (int): a seed for the random number generator
		* ``graph [=None]`` (:py:class:`zen.Graph` or :py:class:`zen.DiGraph`): this is the actual graph instance to populate. It must be
		  empty and its directionality must agree with the value of ``directed``.
	
	**Returns**:
		:py:class:`zen.Graph` or :py:class:`zen.DiGraph`. The graph generated.  If ``directed = True``, then a :py:class:`DiGraph` will be returned.
	
	.. note::
		Source: I. Ispolatov, P. L. Krapivski, and A. Yuryev "Duplication-divergence model of protein interaction network", ??, 2008.
	"""
	seed = kwargs.pop('seed',None)
	directed = kwargs.pop('directed',False)
	graph = kwargs.pop('graph',None)
	
	if graph is not None:
		if len(graph) > 0:
			raise ZenException, 'the graph must be empty, if provided'
		if graph.is_directed() != directed:
			raise ZenException, 'graph and directed arguments must agree'
	
	if len(kwargs) > 0:
		raise ZenException, 'Unknown arguments: %s' % ', '.join(kwargs.keys())
		
	if seed is None:
		seed = -1
		
	# initialize the random number generator
	if seed >= 0:
		random.seed(seed)
			
	if type(n) != int:
		raise ZenException, 'Parameter n must be an integer'
	if type(s) != float and type(s) != int and type(s) != double:
		print type(s)
		raise ZenException, 'Parameter s must be a float, double, or an int'
		
	G = graph
	if graph is None:
		if directed:
			G = DiGraph()
		else:
			G = Graph()
		
	# initialize the graph with two connected nodes
	G.add_edge(0,1)
		
	# build the rest of the graph
	i = 2
	while i < n:
		
		# pick an existing node to copy
		cn_seed = random.randint(0,100000)
		u = choose_node(G,seed=cn_seed)
		
		#####
		# create the partial copy
		G.add_node(i)
		
		# copy edges
		if not directed:
			for w in G.neighbors(u):
				if random.random() <= s:
					G.add_edge(i,w)
		else:
			for w in G.in_neighbors(u):
				if random.random() <= s:
					G.add_edge(w,i)
			for w in G.out_neighbors(u):
				if random.random() <= s:
					G.add_edge(i,w)
					
		# if the node doesn't have any connections, then ditch it
		if G.degree(i) == 0:
			G.rm_node(i)
		else:
			i += 1
					
	# done!
	return G
		
####
# TODO: Implement duplication_divergence_wagner

###
# TODO: Implement duplication_mutation

if __name__ == '__main__':
	from zen.drawing import UbigraphRenderer
	G = DiGraph()
	ur = UbigraphRenderer('http://localhost:20738/RPC2',event_delay=1,graph=G)
	
	G = duplication_divergence_iky(10, 0.4, directed=True, graph=G)
