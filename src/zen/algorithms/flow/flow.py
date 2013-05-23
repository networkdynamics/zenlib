"""
This module implements min_cut, min_cut_, min_cut_set and min_cut_set_

__author__ = James McCorriston
"""

import zen as z

"""
This function returns the min-cut/max flow of the graph G with source node s and sink node t.
The parameter capacity can be set to 'unit' or 'weight' which determines whether edges in the
graph will have capacity 1 or capacity equal to their weight, respectively. If check_graph is
set to true, then the function checks to make sure there are no negative edge weights if the 
capacity is set to 'weight' mode.
"""
def min_cut(G, s , t, capacity='unit', check_graph=True):
	if check_graph and capacity=='weight':
		for eidx, w in G.edges_(weight=True):
			if w < 0:
				raise ZenException, 'min_cut_ only supports non-negative edge weights;'\
					' edge with id %s has weight %s;' % (eidx, w)
	
	return min_cut_(G, G.node_idx(s), G.node_idx(t), capacity)

"""
This function is identical to min_cut except it receives the indeces of the source and sink
nodes instead of the node object. Also there is no check for negative edge weights.
"""
def min_cut_(G, sidx, tidx, capacity='unit'):
	if type(G) is not z.DiGraph:
		raise ZenException, 'min_cut_ only supports DiGraph;'\
				' found %s.' % type(G)
	
	if G.node_object(sidx) == None:
		raise ZenException, 'the source node is not in the graph;'
	else:
		s = G.node_object(sidx)
	if G.node_object(tidx) == None:
		raise ZenException, 'the sink node is not in the graph;'
	else:
		t = G.node_object(tidx)
	if capacity != 'unit' and capacity != 'weight':
		raise ZenException, 'capacity must either be \'unit\' or \'weight\''
	
	#copies the inputted graph, the copy will be used as the residual capacities graph
	residual_capacities = G.copy()
	residual_capacities = create_residual_digraph(residual_capacities, capacity)
	#creates a dummy graph that stores the source node and all of its outgoing edges (to be used for flow calculation)
	dg = z.DiGraph()
	dg.add_node(s)
	for u,v,w in residual_capacities.out_edges(s, weight=True):
		dg.add_node(v)
		dg.add_edge(u, v, weight=w)

	try:
		#execute the Ford-Fulkerson algorithm to compute the residual capacities graph
		residual_capacities = ford_fulkerson(residual_capacities, s, t, capacity)
	except:
		return float('inf')

	#the flow through each edge is equal to the original capacity - the residual capacity
	flows = result_flow(dg, residual_capacities)
	return sum(flows[u][v] for u,v in dg.out_edges(s))

"""
This function returns the min-cut set of the graph G by first using the Ford-Fulkerson
algorithm to create a residual graph. Any node n that has a walk from s to n, excluding
edges of weight 0, is labeled 'A' and all unreachable nodes are labeled 'B'. The edge set
between the subset of nodes labeled 'A' and the subset labeled 'B' is the min-cut set.
The edge set is returned as a list of (src, tgt) tuples.
"""
def min_cut_set(G, s, t, capacity='unit', check_graph=True):
	if check_graph:
		for eidx, w in G.edges_(weight=True):
			if w < 0:
				raise ZenException, 'min_cut_set only supports non-negative edge weights;'\
					' edge with id %s has weight %s;' % (eidx, w)
	
	edge_set_idxs = min_cut_set_(G, G.node_idx(s), G.node_idx(t), capacity)

	edge_set = []
	#convert the list of edge indexes to a list of (src, tgt) tuples
	for i in edge_set_idxs:
		edge_set = edge_set + [(G.src(i), G.tgt(i))]
	return edge_set
		

"""
This function returns the min-cut set of the graph G by first using the Ford-Fulkerson
algorithm to create a residual graph. Any node n that has a walk from s to n, excluding
edges of weight 0, is labeled 'A' and all unreachable nodes are labeled 'B'. The edge set
between the subset of nodes labeled 'A' and the subset labeled 'B' is the min-cut set.
The edge set is returned as a list of edge indexes.
"""	
def min_cut_set_(G, sidx, tidx, capacity='unit'):
	if type(G) is not z.DiGraph:
		raise ZenException, 'min_cut_set_ only supports DiGraph;'\
				' found %s.' % type(G)
	
	if G.node_object(sidx) == None:
		raise ZenException, 'the source node is not in the graph;'
	else:
		s = G.node_object(sidx)
	if G.node_object(tidx) == None:
		raise ZenException, 'the sink node is not in the graph;'
	else:
		t = G.node_object(tidx)
	if capacity != 'unit' and capacity != 'weight':
		raise ZenException, 'capacity must either be \'unit\' or \'weight\''

	residual_capacities = G.copy()
	residual_capacities = create_residual_digraph(residual_capacities, capacity)
	
	try:
		residual_capacities = ford_fulkerson(residual_capacities, s, t, capacity)
	except:
		return G.out_edges(s)
	
	edge_set = []

	#splits the graph into two subgraphs, 'A' and 'B'
	H = G.copy()
	for node in residual_capacities.nodes():
		if z.algorithms.shortest_path.dijkstra_path(residual_capacities, s, node)[1] == None:
			#nodes not reachable from s
			H.set_node_data(node, 'B')
		else:
			#nodes reachable from s
			H.set_node_data(node, 'A')

	#finds all edges between the two subgraphs
	for node,data in H.nodes(data=True):
		if data=='A':
			#edges from A to B
			for u,v in H.out_edges(node):
				if H.node_data(v) == 'B':
					edge_set = edge_set + [H.edge_idx(u,v)]
			#edges from B to A
			for u,v in H.in_edges(node):
				if H.node_data(u) == 'B':
					edge_set = edge_set + [H.edge_idx(u,v)]
	return edge_set		

#implementation of the Ford-Fulkerson algorithm
def ford_fulkerson(G, s, t, capacity):
	
	path = get_path(G, s, t)
	#while their is an augmenting path, augments flow by the minimum capacity in that path
	while path != None:
		flow = min(G.weight(u,v) for u,v in path)
		if flow == float('infinity'):
			raise ZenException, 'path from s to t with infinite capacity;'
		else:
			for u,v in path:
				if G.has_edge(u, v):
					#updates the capacity of each edge in the path
					G.set_weight(u, v, (G.weight(u, v) - flow))
					#removes edges of 0 capacity
					if G.weight(u,v)<1:
						G.rm_edge(u,v)
				else:
					G.add_edge(u, v, weight=(dg.weight(u,v) - flow))
				if G.has_edge(v, u):
					#updates residual capacity
					G.set_weight(v, u, (G.weight(v, u) + flow))
					#removes edges of 0 capacity
					if G.weight(v,u)<1:
						G.rm_edge(v,u)
				else:
					#adds a residual edge when necessary
					G.add_edge(v, u, weight=flow)
			path = get_path(G, s, t)
	return G
	

#initializes a residual graph, setting all edge weights to 1 if capacity option is 'unit'
def create_residual_digraph(G, capacity):
	residual = z.DiGraph()
	for node in G.nodes():
		residual.add_node(nobj = node)
	if capacity == 'unit':
		for u,v in G.edges():
			residual.add_edge(u, v, weight=1)
	else:
		for u,v,w in G.edges(weight = True):
			residual.add_edge(u, v, weight=w)

	return residual		
	
#returns a dictionary that stores the resulting flow for each edge
def result_flow(G, H):
	flow = dict([(u, {}) for u,v in G.edges()])

	for u,v in G.edges():
		if H.has_edge(u,v):
			#the flow through edge (u,v) is equal to the original capacity of the edge - its residual capactiy
			flow[u][v] = max(0, G.weight(u,v) - H.weight(u,v))
		else:
			#if the edge has capacity 0 in the residual graph, subtraction is not necessary
			flow[u][v] = max(0, G.weight(u,v))
	
	return flow
	 

#returns an array of edges in the form (u,v) representing the shortest path from s to t in G
def get_path(G, s, t):
	#uses Dijkstra's algorithm to find the shortest path
	shortest_path = z.algorithms.shortest_path.dijkstra_path(G, s, t)[1]
	if shortest_path == None:
		path = None
	else:
		path = []
		for i in xrange(len(shortest_path)-1):
			#converts the shortest path from a list of nodes to a list of (u,v) edges
			path = path + [(shortest_path[i], shortest_path[i+1])]
	return path
