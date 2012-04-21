#cython: embedsignature=True

"""
The ``zen.algorithms.centrality`` module provides a number of centrality measures on graphs.  Currently supported measures are:

	* Betweenness centrality (:py:func:`betweenness_centrality` and :py:func:`betweenness_centrality_`)
	* Eigenvector centrality (:py:func:`eigenvector_centrality` and :py:func:`eigenvector_centrality_`)
	
.. note::
	Each measure has two functions.  ``<centrality_fxn>`` accepts node objects as the identifiers for nodes.  ``<centrality_measure>_`` (note the underscore) accepts node
	indices as the identifiers for nodes.  The node index-based function has less overhead since node indices are directly memory references to the nodes of interest.  When
	node objects are used, a dictionary lookup is required to find the node identified.
	
Betweenness centrality
----------------------

`Betweenness centrality <http://en.wikipedia.org/wiki/Betweenness_centrality>`_ ranks a node (or edge) according to the number of shortest paths traversing the network
pass through it.  The more paths that pass through it, the higher its rank.

.. note::
	The algorithm used is from Ulrik Brandes, `A Faster Algorithm for Betweenness Centrality <http://www.inf.uni-konstanz.de/algo/publications/b-fabc-01.pdf>`_. 
	Journal of Mathematical Sociology 25(2):163-177, 2001.

.. autofunction:: zen.algorithms.centrality.betweenness_centrality

.. autofunction:: zen.algorithms.centrality.betweenness_centrality_


Eigenvector centrality
----------------------

`Eigenvector centrality <http://en.wikipedia.org/wiki/Centrality#Eigenvector_centrality>`_ calculates a node's centrality as a function of the centrality of its neighbors.

Because this definition is intrinsically recursive, not surprisingly, the best known method for evaluating eigenvector centrality is an iterative algorithm called the `power method <http://en.wikipedia.org/wiki/Power_iteration>`_.  This method finds the eigenvector for the largest eigenvalue of the adjacency matrix of G. A drawback of this approach is that
its convergence time to the correct solution can be long.

.. note::
	Be aware that the power iteration method provides no guarantee of convergence.

.. autofunction:: zen.algorithms.centrality.eigenvector_centrality

.. autofunction:: zen.algorithms.centrality.eigenvector_centrality_

"""

__all__ = [	'betweenness_centrality',
			'betweenness_centrality_',
			'eigenvector_centrality',
			'eigenvector_centrality_']

import heapq

cimport numpy as np
import numpy as np
from zen.digraph cimport *
from zen.graph cimport *
from exceptions import *

cpdef brandes_betweenness(G,bint normalized=True,bint weighted=False):
	"""
	Compute betweenness centrality for all nodes in the network ``G``.

	**Args**:
	
		* ``normalized [=True]`` (boolean): if ``True``, then the betweenness values are normalized by ``b=b/(N-1)(N-2)`` where ``N`` is the number of nodes in ``G``.
		* ``weighted [=False]`` (boolean): if ``True``, then the shortest path weights are incorporated into the betweenness calculation.

	**Returns**:
		:py:class:`dict`, B. B[n] is the betweenness score for node with object ``n``.

	.. note::
		The algorithm used is from Ulrik Brandes,`A Faster Algorithm for Betweenness Centrality <http://www.inf.uni-konstanz.de/algo/publications/b-fabc-01.pdf>`_. 
		Journal of Mathematical Sociology 25(2):163-177, 2001.
	"""
	B_ = brandes_betweenness_(G,normalized,weighted)
	
	B = {}
	for nidx,nobj in G.nodes_iter_(obj=True):
		B[nobj] = B_[nidx]
	
	return B
	
cpdef betweenness_centrality(G,bint normalized=True,bint weighted=False):
	"""
	Compute betweenness centrality for all nodes in the network ``G``.

	**Args**:

		* ``G`` (:py:class:`zen.Graph` or :py:class:`zen.DiGraph`): the graph to compute the betweenness measure on.
		* ``normalized [=True]`` (boolean): if ``True``, then the betweenness values are normalized by ``b=b/(N-1)(N-2)`` where ``N`` is the number of nodes in ``G``.
		* ``weighted [=False]`` (boolean): if ``True``, then the shortest path weights are incorporated into the betweenness calculation.

	**Returns**:
		:py:class:`dict`, ``B``. B[n] is the betweenness score for node with object ``n``.
	"""
	return brandes_betweenness(G,normalized,weighted)

cpdef brandes_betweenness_(G,bint normalized=True,bint weighted=False):
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

cpdef betweenness_centrality_(G,bint normalized=True,bint weighted=False):
	"""
	Compute betweenness centrality for all nodes in the network ``G``.

	**Args**:

		* ``G`` (:py:class:`zen.Graph` or :py:class:`zen.DiGraph`): the graph to compute the betweenness measure on.
		* ``normalized [=True]`` (boolean): if ``True``, then the betweenness values are normalized by ``b=b/(N-1)(N-2)`` where ``N`` is the number of nodes in ``G``.
		* ``weighted [=False]`` (boolean): if ``True``, then the shortest path weights are incorporated into the betweenness calculation.

	**Returns**:
		1-D ``numpy.ndarray``, ``B``. ``B[i]`` is the betweenness score for node with index ``i``.
	"""
	return brandes_betweenness_(G,normalized,weighted)

cdef __brandes_betweenness_udir(Graph G,bint normalized,bint weighted):
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

cdef __brandes_betweenness_dir(DiGraph G,bint normalized,bint weighted):
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

cpdef eigenvector_centrality(G,max_iter=100,tol=1.0e-6,bint weighted=False):
	"""
	Calculate the eigenvector centrality for all nodes in graph ``G``.

	**Args**:
		
		* ``G`` (:py:class:`zen.Graph` or :py:class:`zen.DiGraph`): the graph to compute centrality measure on.
		* ``max_iter [=100]`` (int): is the maximum number of iterations to perform.
		* ``tol [=1.0e-6]`` (float): is the error tolerance used to check convergence in the power method iteration.
		  If the error falls below ``tol``, then the method terminates.
		* ``weighted [=False]`` (boolean): if ``True``, then the weights of edges are incorporated into the centrality calculation.
	
	**Returns**:
		:py:class:`dict`, ``C``.  ``C[n]`` is the centrality value for node with object ``n``.
		
	**Raises**:
		:py:exc:`zen.ZenException`: if the method didn't converge in ``max_iter`` iterations.
	"""
	C_ = eigenvector_centrality_(G,max_iter,tol,weighted)
	
	C = {}
	for nidx,nobj in G.nodes_iter_(obj=True):
		C[nobj] = C_[nidx]
	
	return C

cpdef eigenvector_centrality_(G,max_iter=100,tol=1.0e-6,bint weighted=False):
	"""
	Calculate the eigenvector centrality for all nodes in graph ``G``.

	**Args**:
	
		* ``G`` (:py:class:`zen.Graph` or :py:class:`zen.DiGraph`): the graph to compute centrality measure on.
		* ``max_iter [=100]`` (int): is the maximum number of iterations to perform.
		* ``tol [=1.0e-6]`` (float): is the error tolerance used to check convergence in the power method iteration.
		  If the error falls below ``tol``, then the method terminates.
		* ``weighted [=False]`` (boolean): if ``True``, then the weights of edges are incorporated into the centrality calculation.

	**Returns**:
		1-D ``numpy.ndarray``, ``C``. ``C[i]`` is the eigenvector centrality for node with index ``i``.
		
	**Raises**:
		:py:exc:`zen.ZenException`: if the method didn't converge in ``max_iter`` iterations.
	"""
	if type(G) == Graph:
		return __eigenvector_undir_(<Graph> G,max_iter,tol,weighted)
	elif type(G) == DiGraph:
		return __eigenvector_dir_(<DiGraph> G,max_iter,tol,weighted)
	else:
		raise ZenException, 'Unknown graph type: %s' % type(G)

cdef __eigenvector_undir_(Graph G,int max_iter,float tol,bint weighted):
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
	
cdef __eigenvector_dir_(DiGraph G,int max_iter,float tol,bint weighted):
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