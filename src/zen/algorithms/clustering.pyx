"""
The ``zen.algorithms.clustering`` module (available as ``zen.clustering``) implements three measures of the degree of local clustering of nodes
in a network's connectivity:

	* `Global clustering coefficient <http://en.wikipedia.org/wiki/Clustering_coefficient#Global_clustering_coefficient>`_
	* `Local clustering coefficient <http://en.wikipedia.org/wiki/Clustering_coefficient#Local_clustering_coefficient>`_
	* `Network average clustering coefficient <http://en.wikipedia.org/wiki/Clustering_coefficient#Network_average_clustering_coefficient>`_


"""

from zen.digraph cimport DiGraph
from zen.graph cimport Graph
import numpy
from exceptions import *

from cpython cimport bool

__all__ = ['gcc','lcc','tt']

cpdef tt(G):
	"""
	Transitive triples clustering coefficient.
	"""
	if type(G) != DiGraph:
		raise ZenException, 'Graph of type %s not supported' % str(type(G))
		
	cdef DiGraph graph = <DiGraph> G
	
	cdef int num_nodes = 0
	cdef int idx
	cdef int x,i,j,k
	cdef int ni,nj,nk
	cdef int num_vees = 0
	cdef int num_tris = 0

	nodes = graph.nodes_()
	num_nodes = len(nodes)

	for x in range(num_nodes):
		ni = nodes[x]

		#num_vees += graph.node_info[nj].indegree * graph.node_info[nj].outdegree
	
		# check each edge
		for i in range(graph.node_info[ni].outdegree):
			nj = graph.edge_info[graph.node_info[ni].outelist[i]].tgt
			
			if ni == nj:
				continue
				
			for j in range(graph.node_info[nj].outdegree):
				nk = graph.edge_info[graph.node_info[nj].outelist[j]].tgt
				
				if nk == ni or nk == nj:
					continue
					
				num_vees += 1
				
				if G.has_edge_(ni,nk):
					num_tris += 1

	if num_vees == 0:
		return 0
	else:
		return <float> num_tris / <float> num_vees
	
cpdef gcc(G):
	"""
	Compute the global clustering coefficient: the fraction of triangles to vee's in the network.
	"""
	if type(G) == DiGraph:
		return __gcc_directed(<DiGraph> G)
	elif type(G) == Graph:
		return __gcc_undirected(<Graph> G)
	else:
		raise ZenException, 'Graph type %s not supported' % str(type(G))

def lcc(G,**kwargs):
	"""
	Compute the local clustering coefficients for nodes in the network specified.
	By default, a numpy array containing each nodes' clustering coefficient will be
	returned.
	
	Possible other arguments for this function are:
	
	  - nbunch=None: a container holding a set of node descriptors whose clustering
					coefficients should be computed.
	  - nbunch_=None: a container holding a set of node ids whose clustering
					coefficients should be computed.
	  - avg=False: if True, return the average local clustering coefficient for
					the network.  Rather than an array of values, one float will be returned.
					When used with nbunch or nbunch_, this will return the average
					clustering coefficient for just those nodes.
	  - ids=False: if True, the result will be a two-dimensional array in which the first
					column is the clustering coefficient and the second column is the
					associated node id.	
	"""
	if type(G) == DiGraph:
		# parse the results...		
		avg = kwargs.pop('avg',False)
		ids = kwargs.pop('ids',False)
		if avg and ids:
			raise ZenException, 'Both ids and avg cannot be given'
		
		nbunch = kwargs.pop('nbunch',None)
		nbunch_ = kwargs.pop('nbunch_',None)
		if nbunch != None and nbunch_ != None:
			raise ZenException, 'Both nbunch and nbunch_ cannot be given'
				
		# if there are extra arguments...
		if len(kwargs) != 0:
			raise ZenException, 'Unknown arguments: %s' % ','.join(kwargs.keys())
			
		return __lcc_directed(<DiGraph>G,nbunch,nbunch_,avg,ids)
	elif type(G) == Graph:
		# parse the results...
		avg = kwargs.pop('avg',False)
		ids = kwargs.pop('ids',False)
		if avg and ids:
			raise ZenException, 'Both ids and avg cannot be given'
		
		nbunch = kwargs.pop('nbunch',None)
		nbunch_ = kwargs.pop('nbunch_',None)
		if nbunch != None and nbunch_ != None:
			raise ZenException, 'Both nbunch and nbunch_ cannot be given'
				
		# if there are extra arguments...
		if len(kwargs) != 0:
			raise ZenException, 'Unknown arguments: %s' % ','.join(kwargs.keys())
			
		return __lcc_undirected(<Graph>G,nbunch,nbunch_,avg,ids)
	else:
		raise ZenException, 'Unsupported graph type: %s' % str(type(G))

cpdef float __gcc_directed(DiGraph G):
	cdef int i,j,k
	cdef int ni,nj,nk,nl
	cdef int num_vees = 0
	cdef int num_tris = 0
	cdef int num_notself
		
	for ni in range(G.next_node_idx):
		
		if not G.node_info[ni].exists:
			continue
		
		num_notself = 0
	
		# check each edge
		for i in range(G.node_info[ni].outdegree):
			nj = G.edge_info[G.node_info[ni].outelist[i]].tgt
			
			if nj == ni:
				continue
			else:
				num_notself += 1
				
			for j in range(G.node_info[nj].outdegree):
				nk = G.edge_info[G.node_info[nj].outelist[j]].tgt
				for k in range(G.node_info[nk].indegree):
					nl = G.edge_info[G.node_info[nk].inelist[k]].src
					if nl == ni:
						num_tris += 1
						
		num_vees += num_notself * (num_notself - 1)
		
	if num_vees == 0:
		return 0
	else:
		return <float> num_tris / <float> num_vees

cpdef float __gcc_undirected(Graph G):
	cdef int idx
	cdef int i,j,k
	cdef int ni,nj,nk,nl
	cdef int num_vees = 0
	cdef int num_tris = 0
	cdef int num_nonself = 0

	# count the number of unique triangles (regardless of node ordering)
	for ni in range(G.next_node_idx):
		
		if not G.node_info[ni].exists:
			continue
		
		# loop over all nodes adjacent to ni
		num_nonself = 0
		for i in range(G.node_info[ni].degree):
			nj = G.endpoint_(G.node_info[ni].elist[i],ni)
			
			# keep track of how many of ni's edges are non-self loops.  This 
			# is used to compute the number of V's
			if nj != ni:
				num_nonself += 1
				
			if nj <= ni:
				continue
				
			# loop over all nodes adjacent to nj
			for j in range(G.node_info[nj].degree):
				nk = G.endpoint_(G.node_info[nj].elist[j],nj)
				
				if nk <= nj:
					continue
					
				for k in range(G.node_info[nk].degree):
					nl = G.endpoint_(G.node_info[nk].elist[k],nk)
					if nl == ni:
						num_tris += 1
						
		num_vees += (num_nonself * (num_nonself - 1)) / 2
			
	if num_vees == 0:
		return 0
	else:
		return <float> (3 * num_tris) / <float> num_vees

cpdef __wcc_directed(DiGraph G,nbunch,nbunch_,bool avg,bool ids,bool reverse):
	"""
	Compute the weak correlation coefficient for the directed graph G.
	
	Note that the check for nbunch and nbunch_ being both set and the check for 
	both avg and ids being set is done in wcc(...).
	"""
	cdef int num_nodes = 0
	cdef int idx
	cdef int i,j,k
	cdef int ni,nj,nk
	cdef bool to_ids = False
	cdef bool success = False
	cdef int completed_nodes = 0
	
	nodes_iter = None
	
	if nbunch is not None:
		to_ids = True
		nodes_iter = iter(nbunch)
		num_nodes = len(nbunch)
	elif nbunch_ is not None:
		nodes_iter = iter(nbunch_)
		num_nodes = len(nbunch_)
	else:
		nodes_iter = G.nodes_iter_()
		num_nodes = len(G)
		
	# allocate a numpy array for the results
	idx = 0
	result = None
	#result = numpy.empty( (num_nodes,ndims), numpy.float)
	if ids:
		result = numpy.empty( (num_nodes,2), numpy.float)
	else:
		result = numpy.empty(num_nodes, numpy.float)
	
	if not reverse:
		for nidx in nodes_iter:
			# translate a node descriptor to ID if necessary
			if to_ids:
				nidx = G.node_idx(nidx)
			ni = nidx
	
			# check each edge
			completed_nodes = 0
			for i in range(G.node_info[ni].outdegree):
				nj = G.edge_info[G.node_info[ni].outelist[i]].tgt
				success = False
				for j in range(G.node_info[nj].outdegree):
					nk = G.edge_info[G.node_info[nj].outelist[j]].tgt
					for k in range(G.node_info[nk].indegree):
						if G.edge_info[G.node_info[nk].inelist[k]].src == ni:
							success = True
							completed_nodes += 1
							break
					if success:
						break
	
			# compute the wcc for this node
			if ids is True:
				if G.node_info[ni].outdegree == 0:
					result[idx,0] = 0.0
				else:
					result[idx,0] = <float> completed_nodes / <float> G.node_info[ni].outdegree
				result[idx,1] = ni
			else:
				if G.node_info[ni].outdegree == 0:
					result[idx] = 0.0
				else:
					result[idx] = <float> completed_nodes / <float> G.node_info[ni].outdegree
			idx += 1
	else:
		for nidx in nodes_iter:
			# translate a node descriptor to ID if necessary
			if to_ids:
				nidx = G.node_idx(nidx)
			ni = nidx

			# check each edge
			completed_nodes = 0
			for i in range(G.node_info[ni].indegree):
				nj = G.edge_info[G.node_info[ni].inelist[i]].src
				success = False
				for j in range(G.node_info[nj].outdegree):
					nk = G.edge_info[G.node_info[nj].outelist[j]].tgt
					for k in range(G.node_info[nk].outdegree):
						if G.edge_info[G.node_info[nk].outelist[k]].tgt == ni:
							success = True
							completed_nodes += 1
							break
					if success:
						break

			# compute the wcc for this node
			if ids is True:
				if G.node_info[ni].indegree == 0:
					result[idx,0] = 0.0
				else:
					result[idx,0] = <float> completed_nodes / <float> G.node_info[ni].indegree
				result[idx,1] = ni
			else:
				if G.node_info[ni].indegree == 0:
					result[idx] = 0.0
				else:
					result[idx] = <float> completed_nodes / <float> G.node_info[ni].indegree
			idx += 1
		
	# return the result
	if avg is True:
		return numpy.average(result)
	else:
		return result
		
cpdef __lcc_directed(DiGraph G,nbunch,nbunch_,bool avg,bool ids):
	"""
	Compute the local clustering coefficient for the directed graph G.

	Note that the check for nbunch and nbunch_ being both set and the check for 
	both avg and ids being set is done in icc(...).
	"""
	cdef int num_nodes = 0
	cdef int idx
	cdef int i,j,k
	cdef int ni,nj,nk
	cdef bool to_ids = False
	cdef bool success = False
	cdef int completed_nodes = 0
	cdef int num_vees = 0
	cdef int num_tris = 0

	nodes_iter = None

	if nbunch is not None:
		to_ids = True
		nodes_iter = iter(nbunch)
		num_nodes = len(nbunch)
	elif nbunch_ is not None:
		nodes_iter = iter(nbunch_)
		num_nodes = len(nbunch_)
	else:
		nodes_iter = G.nodes_iter_()
		num_nodes = len(G)

	# allocate a numpy array for the results
	idx = 0
	result = None
	#result = numpy.empty( (num_nodes,ndims), numpy.float)
	if ids:
		result = numpy.empty( (num_nodes,2), numpy.float)
	else:
		result = numpy.empty(num_nodes, numpy.float)

	for nidx in nodes_iter:
		# translate a node descriptor to ID if necessary
		if to_ids:
			nidx = G.node_idx(nidx)
		ni = nidx

		num_vees = 0 #G.node_info[ni].outdegree * (G.node_info[ni].outdegree-1)
		num_tris = 0
		
		# check each edge
		for i in range(G.node_info[ni].outdegree):
			nj = G.edge_info[G.node_info[ni].outelist[i]].tgt
			
			if nj == ni:
				continue
				
			for j in range(G.node_info[ni].outdegree):				
				nk = G.edge_info[G.node_info[ni].outelist[j]].tgt
				
				if nk == ni or nk == nj:
					continue
					
				num_vees += 1
				if G.has_edge_(nj,nk):
					num_tris += 1

		# compute the icc for this node
		if ids is True:
			if num_vees == 0:
				result[idx,0] = 0.0
			else:
				result[idx,0] = <float> num_tris / <float> num_vees
			result[idx,1] = ni
		else:
			if num_vees == 0:
				result[idx] = 0.0
			else:
				result[idx] = <float> num_tris / <float> num_vees
		idx += 1

	# return the result
	if avg is True:
		return numpy.average(result)
	else:
		return result
		
cpdef __lcc_undirected(Graph G,nbunch,nbunch_,bool avg,bool ids):
	"""
	Compute the local clustering coefficient for the directed graph G.

	Note that the check for nbunch and nbunch_ being both set and the check for 
	both avg and ids being set is done in icc(...).
	"""
	cdef int num_nodes = 0
	cdef int idx
	cdef int i,j,k
	cdef int ni,nj,nk
	cdef bool to_ids = False
	cdef bool success = False
	cdef int completed_nodes = 0
	cdef int num_vees = 0
	cdef int num_tris = 0

	nodes_iter = None

	if nbunch is not None:
		to_ids = True
		nodes_iter = iter(nbunch)
		num_nodes = len(nbunch)
	elif nbunch_ is not None:
		nodes_iter = iter(nbunch_)
		num_nodes = len(nbunch_)
	else:
		nodes_iter = G.nodes_iter_()
		num_nodes = len(G)

	# allocate a numpy array for the results
	idx = 0
	result = None
	#result = numpy.empty( (num_nodes,ndims), numpy.float)
	if ids:
		result = numpy.empty( (num_nodes,2), numpy.float)
	else:
		result = numpy.empty(num_nodes, numpy.float)

	for nidx in nodes_iter:
		# translate a node descriptor to ID if necessary
		if to_ids:
			nidx = G.node_idx(nidx)
		ni = nidx

		num_vees = <int> ( (<float> G.node_info[ni].degree * (G.node_info[ni].degree-1)) / 2.)
		num_tris = 0

		# check each edge
		for i in range(G.node_info[ni].degree):
			if G.edge_info[G.node_info[ni].elist[i]].u == ni:
				nj = G.edge_info[G.node_info[ni].elist[i]].v
			else:
				nj = G.edge_info[G.node_info[ni].elist[i]].u
				
			for j in range(i+1,G.node_info[ni].degree):
				if G.edge_info[G.node_info[ni].elist[j]].u == ni:
					nk = G.edge_info[G.node_info[ni].elist[j]].v
				else:
					nk = G.edge_info[G.node_info[ni].elist[j]].u
				
				if G.has_edge_(nj,nk):
					num_tris += 1
				
		# compute the icc for this node
		if ids is True:
			if num_vees == 0:
				result[idx,0] = 0.0
			else:
				result[idx,0] = <float> num_tris / <float> num_vees
			result[idx,1] = ni
		else:
			if num_vees == 0:
				result[idx] = 0.0
			else:
				result[idx] = <float> num_tris / <float> num_vees
		idx += 1

	# return the result
	if avg is True:
		return numpy.average(result)
	else:
		return result