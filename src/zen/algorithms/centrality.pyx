#cython: embedsignature=True

"""
Betweenness centrality and similar centrality measures.
"""

__all__ = [	'betweenness',
			'betweenness_',
			'eigenvector',
			'eigenvector_']
#,
#		   'betweenness_centrality_source',
#		   'edge_betweenness']

import heapq

cimport numpy as np
import numpy as np
from zen.digraph cimport *
from zen.graph cimport *
from exceptions import *

from cpython cimport bool

cpdef brandes_betweenness(G,bool normalized=True,bool weighted=False):
	"""
	Compute betweenness centrality for all nodes in the network G.  The result is a dictionary, B, where B[n] 
	is the betweenness for node with object n.

	If normalized is True, then the betweenness values are normalized by b=b/(n-1)(n-2) where n is the number of nodes in G.

	If weighted is True, then the edge weights in determining the shortest paths.

	The algorithm used is from Ulrik Brandes,A Faster Algorithm for Betweenness Centrality. Journal of Mathematical Sociology 25(2):163-177, 2001.
		http://www.inf.uni-konstanz.de/algo/publications/b-fabc-01.pdf
	"""
	B_ = brandes_betweenness_(G,normalized,weighted)
	
	B = {}
	for nidx,nobj in G.nodes_iter_(obj=True):
		B[nobj] = B_[nidx]
	
	return B
	
cpdef betweenness(G,bool normalized=True,bool weighted=False):
	return brandes_betweenness(G,normalized,weighted)

cpdef brandes_betweenness_(G,bool normalized=True,bool weighted=False):
	"""
	Compute betweenness centrality for all nodes in the network G.  The result is a numpy array, B, where B[i] 
	is the betweenness of node with index i.

	If normalized is True, then the betweenness values are normalized by b=b/(n-1)(n-2) where n is the number of nodes in G.
   
	If weighted is True, then the edge weights in determining the shortest paths.

	The algorithm used is from Ulrik Brandes,A Faster Algorithm for Betweenness Centrality. Journal of Mathematical Sociology 25(2):163-177, 2001.
		http://www.inf.uni-konstanz.de/algo/publications/b-fabc-01.pdf
	"""
	if type(G) == Graph:
		return __brandes_betweenness_udir(<Graph> G,normalized,weighted)
	elif type(G) == DiGraph:
		return __brandes_betweenness_dir(<DiGraph> G,normalized,weighted)
	else:
		raise ZenException, 'Unknown graph type: %s' % type(G)

cpdef betweenness_(G,bool normalized=True,bool weighted=False):
	return brandes_betweenness_(G,normalized,weighted)

cdef __brandes_betweenness_udir(Graph G,bool normalized,bool weighted):
	cdef np.ndarray[np.float_t, ndim=1] betweenness = np.zeros(G.next_node_idx, np.float)	
	cdef np.ndarray[np.float_t, ndim=1] delta = np.zeros(G.next_node_idx, np.float)
	cdef np.ndarray[np.float_t, ndim=1] D = np.zeros(G.next_node_idx, np.float)
	cdef np.ndarray[np.float_t, ndim=1] sigma = np.zeros(G.next_node_idx, np.float)
	cdef np.ndarray[np.float_t, ndim=1] seen = np.zeros(G.next_node_idx, np.float)
	cdef np.ndarray[np.int_t, ndim=1] Sarray = np.zeros(G.num_nodes, np.int)
	cdef int sidx
	
	cdef int s,v,w,tmp,eidx,i
	cdef float vw_dist,scale
	
	for s in range(G.next_node_idx): #G.nodes_iter_():
		if not G.node_info[s].exists:
			continue
			
		#S=[]
		sidx = 0
		P={}
		for v in range(G.next_node_idx): #G.nodes_iter_():
			if not G.node_info[v].exists:
				continue
			P[v]=[]
			
		sigma.fill(0)
		D.fill(-1)
		sigma[s]=1
		if not weighted:  # use BFS
			D[s]=0
			Q=[s]
			while Q:   # use BFS to find shortest paths
				v=Q.pop(0)
				
				Sarray[sidx] = v
				sidx += 1
				
				for i in range(G.node_info[v].degree):
					w = G.endpoint_(G.node_info[v].elist[i],v)
					
					if D[w] < 0: #w not in D:
						Q.append(w)
						D[w]=D[v]+1
					if D[w]==D[v]+1:   # this is a shortest path, count paths
						sigma[w]=sigma[w]+sigma[v]
						P[w].append(v) # predecessors 
		else:  # use Dijkstra's algorithm for shortest paths,
			   # modified from Eppstein
			push=heapq.heappush
			pop=heapq.heappop
			seen.fill(-1)
			seen[s] = 0
			Q=[]   # use Q as heap with (distance,node id) tuples
			push(Q,(0,s,s))
			while Q:   
				(dist,pred,v)=pop(Q)
				
				if D[v] >= 0:
					continue # already searched this node.
					
				sigma[v]=sigma[v]+sigma[pred] # count paths

				Sarray[sidx] = v
				sidx += 1
				
				D[v] = dist
				#for w,edgedata in G[v].iteritems():
				for i in range(G.node_info[v].degree):
					# get the neighbor
					eidx = G.node_info[v].elist[i]
					w = G.endpoint_(eidx,v)
						
					vw_dist = D[v] + G.weight_(eidx)
					if D[w] < 0 and (seen[w] < 0 or vw_dist < seen[w]):
						seen[w] = vw_dist
						push(Q,(vw_dist,v,w))
						sigma[w]=0
						P[w]=[v]
					elif vw_dist==seen[w]:  # handle equal paths
						sigma[w]=sigma[w]+sigma[v]
						P[w].append(v)


		delta.fill(0)
		for i in range(sidx):
			w = Sarray[sidx-i-1]
			for v in P[w]:
				delta[v]=delta[v]+\
						  (float(sigma[v])/float(sigma[w]))*(1.0+delta[w])
			if w != s:
				betweenness[w]=betweenness[w]+delta[w]
					
	# normalize
	if normalized:
		order=len(betweenness)
		if order <=2:
			return betweenness # no normalization b=0 for all nodes
		scale=1.0/((order-1)*(order-2))
		# for v in betweenness:
		for i in range(G.next_node_idx):
			betweenness[i] *= scale

	# done!
	return betweenness

cdef __brandes_betweenness_dir(DiGraph G,bool normalized,bool weighted):
	"""
	The directed version of this implementation.
	
	The algorithm used is from Ulrik Brandes,A Faster Algorithm for Betweenness Centrality. Journal of Mathematical Sociology 25(2):163-177, 2001.
		http://www.inf.uni-konstanz.de/algo/publications/b-fabc-01.pdf
	"""
	cdef np.ndarray[np.float_t, ndim=1] betweenness = np.zeros(G.next_node_idx, np.float)	
	cdef np.ndarray[np.float_t, ndim=1] delta = np.zeros(G.next_node_idx, np.float)
	cdef np.ndarray[np.float_t, ndim=1] D = np.zeros(G.next_node_idx, np.float)
	cdef np.ndarray[np.float_t, ndim=1] sigma = np.zeros(G.next_node_idx, np.float)
	cdef np.ndarray[np.float_t, ndim=1] seen = np.zeros(G.next_node_idx, np.float)
	cdef np.ndarray[np.int_t, ndim=1] Sarray = np.zeros(G.num_nodes, np.int)
	cdef int sidx

	cdef int s,v,w,tmp,eidx,i
	cdef float vw_dist,scale

	for s in range(G.next_node_idx): #G.nodes_iter_():
		if not G.node_info[s].exists:
			continue

		sidx = 0
		P={}
		for v in range(G.next_node_idx): #G.nodes_iter_():
			if not G.node_info[v].exists:
				continue
			P[v]=[]

		sigma.fill(0)
		D.fill(-1)
		sigma[s]=1
		if not weighted:  # use BFS
			D[s]=0
			Q=[s]
			while Q:   # use BFS to find shortest paths
				v=Q.pop(0)

				Sarray[sidx] = v
				sidx += 1

				for i in range(G.node_info[v].outdegree):
					w = G.edge_info[G.node_info[v].outelist[i]].tgt

					if D[w] < 0: #w not in D:
						Q.append(w)
						D[w]=D[v]+1
					if D[w]==D[v]+1:   # this is a shortest path, count paths
						sigma[w]=sigma[w]+sigma[v]
						P[w].append(v) # predecessors 
		else:  # use Dijkstra's algorithm for shortest paths,
			   # modified from Eppstein
			push=heapq.heappush
			pop=heapq.heappop
			seen.fill(-1)
			seen[s] = 0
			Q=[]   # use Q as heap with (distance,node id) tuples
			push(Q,(0,s,s))
			while Q:   
				(dist,pred,v)=pop(Q)

				if D[v] >= 0:
					continue # already searched this node.

				sigma[v]=sigma[v]+sigma[pred] # count paths

				Sarray[sidx] = v
				sidx += 1

				D[v] = dist
				#for w,edgedata in G[v].iteritems():
				for i in range(G.node_info[v].outdegree):
					# get the neighbor
					eidx = G.node_info[v].outelist[i]
					w = G.edge_info[eidx].tgt

					vw_dist = D[v] + G.weight_(eidx)
					if D[w] < 0 and (seen[w] < 0 or vw_dist < seen[w]):
						seen[w] = vw_dist
						push(Q,(vw_dist,v,w))
						sigma[w]=0
						P[w]=[v]
					elif vw_dist==seen[w]:  # handle equal paths
						sigma[w]=sigma[w]+sigma[v]
						P[w].append(v)


		delta.fill(0)
		for i in range(sidx):
			w = Sarray[sidx-i-1]
			for v in P[w]:
				delta[v]=delta[v]+\
						  (float(sigma[v])/float(sigma[w]))*(1.0+delta[w])
			if w != s:
				betweenness[w]=betweenness[w]+delta[w]

	# normalize
	if normalized:
		order=len(betweenness)
		if order <=2:
			return betweenness # no normalization b=0 for all nodes
		scale=1.0/((order-1)*(order-2))
		# for v in betweenness:
		for i in range(G.next_node_idx):
			betweenness[i] *= scale

	# done!
	return betweenness

### TO CONVERT ######

# def betweenness_centrality_source(G,normalized=True,
# 								  weighted_edges=False,
# 								  sources=None):
# 	"""Compute betweenness centrality for a subgraph.
# 
# 	Enchanced version of the method in centrality module that allows
# 	specifying a list of sources (subgraph).
# 
# 
# 	Parameters
# 	----------
# 	G : graph
# 	  A networkx graph 
# 
# 	normalized : bool, optional
# 	  If True the betweenness values are normalized by b=b/(n-1)(n-2) where
# 	  n is the number of nodes in G.
# 	   
# 	weighted_edges : bool, optional
# 	  Consider the edge weights in determining the shortest paths.
# 	  If False, all edge weights are considered equal.
# 
# 	sources : node list 
# 	  A list of nodes to consider as sources for shortest paths.
# 	  
# 
# 	Returns
# 	-------
# 	nodes : dictionary
# 	   Dictionary of nodes with betweeness centrality as the value.
# 
# 
# 	Notes
# 	-----
# 	See Sec. 4 in 
# 	Ulrik Brandes,
# 	A Faster Algorithm for Betweenness Centrality.
# 	Journal of Mathematical Sociology 25(2):163-177, 2001.
# 	http://www.inf.uni-konstanz.de/algo/publications/b-fabc-01.pdf
# 
# 	This algorithm does not count the endpoints, i.e.
# 	a path from s to t does not contribute to the betweenness of s and t.
# 	"""
# 	if sources == None:
# 		sources = G   # only used to iterate over nodes.
# 
# 	betweenness=dict.fromkeys(G,0.0)
# 	for s in sources:
# 		S,P,D,sigma = _brandes_betweenness_helper(G,s,weighted_edges)
# 
# 		delta=dict.fromkeys(G,0) # unnormalized betweenness
# 		while S:
# 			w=S.pop()
# 			for v in P[w]:
# 				delta[v] += (1.0+delta[w])*sigma[v]/sigma[w] # 1.0 converts all to float
# 			if w == s:
# 				continue
# 			betweenness[w] = betweenness[w] + delta[w]
# 				   
# 	# normalize to size of entire graph
# 	if normalized and G.number_of_edges() > 1:
# 		order=len(betweenness)
# 		scale=1.0/((order-1)*(order-2))
# 		for v in betweenness:
# 			betweenness[v] *= scale
# 
# 	return betweenness
# 
# 
# def edge_betweenness(G,normalized=True,weighted_edges=False,sources=None):
# 	"""Compute betweenness centrality for edges.
# 
# 	Parameters
# 	----------
# 	G : graph
# 	  A networkx graph 
# 
# 	normalized : bool, optional
# 	  If True the betweenness values are normalized by b=b/(n-1)(n-2) where
# 	  n is the number of nodes in G.
# 	   
# 	weighted_edges : bool, optional
# 	  Consider the edge weights in determining the shortest paths.
# 	  If False, all edge weights are considered equal.
# 
# 	sources : node list 
# 	  A list of nodes to consider as sources for shortest paths.
# 	  
# 
# 	Returns
# 	-------
# 	nodes : dictionary
# 	   Dictionary of edges with betweeness centrality as the value.
# 	"""
# 	if sources is None:
# 		sources = G # only used to iterate over nodes
# 
# 	betweenness=dict.fromkeys(G.edges(),0.0)
# 	if G.is_directed():
# 		for s in sources:
# 			S, P, D, sigma =_brandes_betweenness_helper(G,s,weighted_edges)
# 			delta=dict.fromkeys(G,0.0)
# 			while S:
# 				w=S.pop()
# 				for v in P[w]:
# 					edgeFlow = (1.0+delta[w])*sigma[v]/sigma[w] # 1.0 converts all to float
# 					edge = (v,w)
# 					delta[v]		  += edgeFlow
# 					betweenness[edge] += edgeFlow
# 	else:
# 		for s in sources:
# 			S, P, D, sigma =_brandes_betweenness_helper(G,s,weighted_edges)
# 			delta=dict.fromkeys(G,0.0)
# 			while S:
# 				w=S.pop()
# 				for v in P[w]:
# 					edgeFlow = (1.0+delta[w])*sigma[v]/sigma[w] # 1.0 converts all to float
# 					edge = (v,w)
# 					if edge not in betweenness:
# 						edge = (w,v)
# 					delta[v]		  += edgeFlow
# 					betweenness[edge] += edgeFlow
# 
# 	size=len(betweenness)				
# 	if normalized and size > 1:
# 		# normalize to size of entire graph (beware of disconnected components)
# 		scale=1.0/((size-1)*(size-2))
# 		for edge in betweenness:
# 			betweenness[edge] *= scale
# 
# 	return betweenness
# 
# 
# def _brandes_betweenness_helper(G,root,weighted_edges):
# 	"""
# 	Helper for betweenness centrality and edge betweenness centrality.
# 
# 	Runs single-source shortest path from root node.
# 
# 	weighted_edges:: consider edge weights 
# 
# 	Finds::
# 
# 	S=[] list of nodes reached during traversal
# 	P={} predecessors, keyed by child node
# 	D={} distances
# 	sigma={} indexed by node, is the number of paths to root
# 	going through the node
# 	"""
# 	S=[]
# 	P={}
# 	for v in G:
# 		P[v]=[]
# 	sigma=dict.fromkeys(G,0.0)
# 	D={}
# 	sigma[root]=1
# 
# 	if not weighted_edges:  # use BFS
# 		D[root]=0
# 		Q=[root]
# 		while Q:   # use BFS to find shortest paths
# 			v=Q.pop(0)
# 			S.append(v)
# 			for w in G[v]: #  for w in G.adj[v]: # speed hack, exposes internals
# 				if w not in D:
# 					Q.append(w)
# 					D[w]=D[v]+1
# 				if D[w]==D[v]+1:   # this is a shortest path, count paths
# 					sigma[w]=sigma[w]+sigma[v]
# 					P[w].append(v) # predecessors
# 	else:  # use Dijkstra's algorithm for shortest paths,
# 		   # modified from Eppstein
# 		push=heapq.heappush
# 		pop=heapq.heappop
# 		seen = {root:0}
# 		Q=[]   # use Q as heap with (distance,node id) tuples
# 		push(Q,(0,root,root))
# 		while Q:   
# 			(dist,pred,v)=pop(Q)
# 			if v in D:
# 				continue # already searched this node.
# 			sigma[v]=sigma[v]+sigma[pred] # count paths
# 			S.append(v)
# 			D[v] = dist
# 			for w,edgedata in G[v].iteritems(): 
# 				vw_dist = D[v] + edgedata.get('weight',1)
# 				if w not in D and (w not in seen or vw_dist < seen[w]):
# 					seen[w] = vw_dist
# 					sigma[w] = 0
# 					push(Q,(vw_dist,v,w))
# 					P[w]=[v]
# 				elif vw_dist==seen[w]:  # handle equal paths
# 					sigma[w]=sigma[w]+sigma[v]
# 					P[w].append(v)
# 	return S, P, D, sigma

cpdef eigenvector(G,max_iter=100,tol=1.0e-6,bool weighted=False):
	"""
	Return the eigenvector centrality for graph G as a dictionary, C, where
	C[n] is the centrality value for node with object n.

	This method uses the power method to find the eigenvector for the 
	largest eigenvalue of the adjacency matrix of G.

	max_iter is the maximum number of iterations in power method.

	tol is the error tolerance used to check convergence in power method iteration.

	Be aware that the power iteration method provides no guarantee of convergence.
	"""
	C_ = eigenvector_(G,max_iter,tol,weighted)
	
	C = {}
	for nidx,nobj in G.nodes_iter_(obj=True):
		C[nobj] = C_[nidx]
	
	return C

cpdef eigenvector_(G,max_iter=100,tol=1.0e-6,bool weighted=False):
	"""
	Return the eigenvector centrality for graph G as a numpy vector, C, where
	C[i] is the centrality value for node with index i.

	This method uses the power method to find the eigenvector for the 
	largest eigenvalue of the adjacency matrix of G.

	max_iter is the maximum number of iterations in power method.

	tol is the error tolerance used to check convergence in power method iteration.

	Be aware that the power iteration method provides no guarantee of convergence.
	"""
	if type(G) == Graph:
		return __eigenvector_undir_(<Graph> G,max_iter,tol,weighted)
	elif type(G) == DiGraph:
		return __eigenvector_dir_(<DiGraph> G,max_iter,tol,weighted)
	else:
		raise ZenException, 'Unknown graph type: %s' % type(G)

cdef __eigenvector_undir_(Graph G,int max_iter,float tol,bool weighted):
	cdef np.ndarray[np.float_t, ndim=1] x = np.random.random(G.next_node_idx)
	cdef np.ndarray[np.float_t, ndim=1] xlast = np.zeros(G.next_node_idx, np.float)
	cdef np.ndarray[np.float_t, ndim=1] tmp
	cdef float nnodes,s,total,err
	cdef int i,j,n,eidx
	
	# normalize starting vector
	total = 0
	for i in range(G.next_node_idx):
		if G.node_info[i].exists:
			total += x[i]			
	s=1.0/total
	for i in range(G.next_node_idx):
		if G.node_info[i].exists:
			x[i] *= s
			
	nnodes=<float> len(G)
	tol = tol*(<float> len(G))
	
	# make up to max_iter iterations		
	for i in range(max_iter):
		#x=dict.fromkeys(xlast.keys(),0)
		tmp = xlast
		xlast=x
		x = tmp
		x.fill(0)
		
		# do the multiplication y=Ax
		#for n in x:
		for n in range(G.next_node_idx):
			if not G.node_info[n].exists:
				continue
		
			#for nbr in G[n]:
			for j in range(G.node_info[n].degree):
				eidx = G.node_info[n].elist[j]
				nbr = G.endpoint_(eidx,n)
				if weighted:
					x[n]+=xlast[nbr]*G.weight_(eidx) #G[n][nbr].get('weight',1)
				else:
					x[n]+=xlast[nbr]
				
		# normalize vector
		total = 0
		for j in range(G.next_node_idx):
			if G.node_info[j].exists:
				total += x[j]			
		s=1.0/total
		err = 0
		for j in range(G.next_node_idx):
			if G.node_info[j].exists:
				x[j] *= s
				err += abs(x[j]-xlast[j])

		# check convergence			
		if err < tol:
			return x
			
	raise ZenException, 'Eigenvector did not converge'
	
cdef __eigenvector_dir_(DiGraph G,int max_iter,float tol,bool weighted):
	cdef np.ndarray[np.float_t, ndim=1] x = np.random.random(G.next_node_idx)
	cdef np.ndarray[np.float_t, ndim=1] xlast = np.zeros(G.next_node_idx, np.float)
	cdef np.ndarray[np.float_t, ndim=1] tmp
	cdef float nnodes,s,total,err
	cdef int i,j,n,eidx
	
	# normalize starting vector
	total = 0
	for i in range(G.next_node_idx):
		if G.node_info[i].exists:
			total += x[i]			
	s=1.0/total
	for i in range(G.next_node_idx):
		if G.node_info[i].exists:
			x[i] *= s

	nnodes=<float> len(G)
	tol = tol*(<float> len(G))

	# make up to max_iter iterations		
	for i in range(max_iter):
		#x=dict.fromkeys(xlast.keys(),0)
		tmp = xlast
		xlast=x
		x = tmp
		x.fill(0)

		# do the multiplication y=Ax
		#for n in x:
		for n in range(G.next_node_idx):
			if not G.node_info[n].exists:
				continue

			#for nbr in G[n]:
			for j in range(G.node_info[n].outdegree):
				eidx = G.node_info[n].outelist[j]
				nbr = G.edge_info[eidx].tgt
				if weighted:
					x[n]+=xlast[nbr]*G.weight_(eidx) #G[n][nbr].get('weight',1)
				else:
					x[n]+=xlast[nbr]

		# normalize vector
		total = 0
		for j in range(G.next_node_idx):
			if G.node_info[j].exists:
				total += x[j]			
		s=1.0/total
		err = 0
		for j in range(G.next_node_idx):
			if G.node_info[j].exists:
				x[j] *= s
				err += abs(x[j]-xlast[j])

		# check convergence			
		if err < tol:
			return x

	raise ZenException, 'Eigenvector did not converge'