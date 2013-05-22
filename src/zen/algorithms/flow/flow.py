"""
This module implements min_cut, min_cut_, min_cut_set and min_cut_set_

__author__ = James McCorriston
"""

import zen as z

def min_cut(G, s , t, capacity='unit'):
	if type(G) is not z.DiGraph:
		raise ZenException, 'min_cut only supports DiGraph;'\
				' found %s.' % type(G)
	
	if s not in G:
		raise ZenException, 'the source node is not in the graph;'
	if t not in G:
		raise ZenException, 'the sink node is not in the graph;'
	
	if capacity != 'unit' and capacity != 'weight':
		raise ZenException, 'capacity must either be \'unit\' or \'weight\''

	for eidx, w in G.edges_(weight=True):
		if w < 0:
			raise ZenException, 'min_cut only supports non-negative edge weights;'\
				' edge with id %s has weight %s;' % (eidx, w)
	residual_capacities = G.copy()
	residual_capacities = create_residual_digraph(residual_capacities, capacity)
	dg = z.DiGraph()
	dg.add_node(s)
	for u,v,w in residual_capacities.out_edges(s, weight=True):
		dg.add_node(v)
		dg.add_edge(u, v, weight=w)

	try:
		print 'test'
		residual_capacities = ford_fulkerson(residual_capacities, s, t, capacity)
	except:
		return float('inf')

	flows = result_flow(dg, residual_capacities)
	return sum(flows[u][v] for u,v in dg.out_edges(s))

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

	for eidx, w in G.edges_(weight=True):
		if w < 0:
			raise ZenException, 'min_cut_ only supports non-negative edge weights;'\
				' edge with id %s has weight %s;' % (eidx, w)
	
	residual_capacities = G.copy()
	residual_capacities = create_residual_digraph(residual_capacities, capacity)
	dg = z.DiGraph()
	dg.add_node(s)
	for u,v,w in residual_capacities.out_edges(s, weight=True):
		dg.add_node(v)
		dg.add_edge(u, v, weight=w)

	try:
		residual_capacities = ford_fulkerson(residual_capacities, s, t, capacity)
	except:
		return float('inf')

	flows = result_flow(dg, residual_capacities)
	return sum(flows[u][v] for u,v in dg.out_edges(s))

def min_cut_set(G, s, t, capacity='unit'):
	if type(G) is not z.DiGraph:
		raise ZenException, 'min_cut_set only supports DiGraph;'\
				' found %s.' % type(G)
	
	if s not in G:
		raise ZenException, 'the source node is not in the graph;'
	if t not in G:
		raise ZenException, 'the sink node is not in the graph;'
	
	if capacity != 'unit' and capacity != 'weight':
		raise ZenException, 'capacity must either be \'unit\' or \'weight\''

	for eidx, w in G.edges_(weight=True):
		if w < 0:
			raise ZenException, 'min_cut_set only supports non-negative edge weights;'\
				' edge with id %s has weight %s;' % (eidx, w)

	residual_capacities = G.copy()
	residual_capacities = create_residual_digraph(residual_capacities, capacity)
	
	try:
		residual_capacities = ford_fulkerson(residual_capacities, s, t, capacity)
	except:
		return G.out_edges(s)
	
	edge_set = []

	H = z.DiGraph()
	for node in residual_capacities.nodes():
		if z.algorithms.shortest_path.dijkstra_path(residual_capacities, s, node)[1] == None:
			G.set_node_data(node, 'B')
		else:
			G.set_node_data(node, 'A')

	for node,data in G.nodes(data=True):
		if data=='A':
			for u,v in G.out_edges(node):
				d = G.node_data(v)
				if d=='B':
					edge_set = edge_set + [(u,v)]
			for u,v in G.in_edges(node):
				d = G.node_data(u)
				if d=='B':
					edge_set = edge_set + [(u,v)]
	return edge_set
			
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

	for eidx, w in G.edges_(weight=True):
		if w < 0:
			raise ZenException, 'min_cut_set_ only supports non-negative edge weights;'\
				' edge with id %s has weight %s;' % (eidx, w)

	residual_capacities = G.copy()
	residual_capacities = create_residual_digraph(residual_capacities, capacity)
	
	try:
		residual_capacities = ford_fulkerson(residual_capacities, s, t, capacity)
	except:
		return G.out_edges(s)
	
	edge_set = []

	H = z.DiGraph()
	for node in residual_capacities.nodes():
		if z.algorithms.shortest_path.dijkstra_path(residual_capacities, s, node)[1] == None:
			G.set_node_data(node, 'B')
		else:
			G.set_node_data(node, 'A')

	for node,data in G.nodes(data=True):
		if data=='A':
			for u,v in G.out_edges(node):
				d = G.node_data(v)
				if d=='B':
					edge_set = edge_set + [(u,v)]
			for u,v in G.in_edges(node):
				d = G.node_data(u)
				if d=='B':
					edge_set = edge_set + [(u,v)]
	return edge_set		

def ford_fulkerson(G, s, t, capacity):
	
	path = get_path(G, s, t)
	while path != None:
		flow = min(G.weight(u,v) for u,v in path)
		if flow == float('infinity'):
			raise ZenException, 'path from s to t with infinite capacity;'
		else:
			for u,v in path:
				if G.has_edge(u, v):
					G.set_weight(u, v, (G.weight(u, v) - flow))
					if G.weight(u,v)<1:
						G.rm_edge(u,v)
				else:
					G.add_edge(u, v, weight=(dg.weight(u,v) - flow))
				if G.has_edge(v, u):
					G.set_weight(v, u, (G.weight(v, u) + flow))
					if G.weight(v,u)<1:
						G.rm_edge(v,u)
				else:
					G.add_edge(v, u, weight=flow)
			path = get_path(G, s, t)
	return G
	

#initializes a residual graph
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
			flow[u][v] = max(0, G.weight(u,v) - H.weight(u,v))
		else:
			flow[u][v] = max(0, G.weight(u,v))
	
	return flow
	 

#returns an array of edges representing the shortest path from s to t in G
def get_path(G, s, t):
	shortest_path = z.algorithms.shortest_path.dijkstra_path(G, s, t)[1]
	if shortest_path == None:
		path = None
	else:
		path = []
		for i in xrange(len(shortest_path)-1):
			path = path + [(shortest_path[i], shortest_path[i+1])]
	return path


#cycles, negative weights, infinite capacities

