#cython: embedsignature=True

"""
Various clustering measures.
"""
from zen.digraph cimport DiGraph
from zen.graph cimport Graph
import numpy
from exceptions import *

from cpython cimport bool

__all__ = ['overall','individual','weak','tt']

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
	
cpdef overall(G):
	"""
	Compute the overall clustering coefficient: the fraction of triangles to vee's in the network.
	"""
	if type(G) == DiGraph:
		return __occ_directed(<DiGraph> G)
	else:
		raise ZenException, 'Graph type %s not supported' % str(type(G))

def local(G,**kwargs):
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

# def weak(G,**kwargs):
# 	"""
# 	Compute the weak clustering coefficients for nodes in the network specified.
# 	By default, a numpy array containing each nodes' weak clustering coefficient will
# 	be returned.
# 	
# 	Possible other arguments for this function are:
# 	
# 	  - nbunch=None: a container holding a set of node descriptors whose clustering
# 					coefficients should be computed.
# 	  - nbunch_=None: a container holding a set of node ids whose clustering
# 					coefficients should be computed.
# 	  - avg=False: if True, return the average weak clustering coefficient for
# 					the network.  Rather than an array of values, one float will be returned.
# 					When used with nbunch or nbunch_, this will return the average weak
# 					clustering coefficient for just those nodes.
# 	  - ids=False: if True, the result will be a two-dimensional array in which the first
# 					column is the weak clustering coefficient and the second column is the
# 					associated node id.
# 	  - reverse=False: if True, then in-neighbors will be considered rather than out-neighbors.
# 	"""
# 	if type(G) == DiGraph:
# 		# parse the results...
# 		avg = False
# 		ids = False
# 		nbunch = None
# 		nbunch_ = None
# 		reverse = False
# 		
# 		if 'avg' in kwargs:
# 			avg = kwargs['avg']
# 			del kwargs['avg']
# 		if 'ids' in kwargs:
# 			ids = kwargs['ids']
# 			del kwargs['ids']
# 			if avg and ids:
# 				raise Exception, 'Both ids and avg cannot be given'
# 		if 'nbunch' in kwargs:
# 			nbunch = kwargs['nbunch']
# 			del kwargs['nbunch']
# 		if 'nbunch_' in kwargs:
# 			nbunch_ = kwargs['nbunch_']
# 			del kwargs['nbunch_']
# 			if nbunch != None and nbunch_ != None:
# 				raise Exception, 'Both nbunch and nbunch_ cannot be given'
# 		if 'reverse' in kwargs:
# 			reverse = kwargs['reverse']	
# 			del kwargs['reverse']
# 		if len(kwargs) != 0:
# 			raise ZenException, 'Unknown arguments: %s' % ','.join(kwargs.keys())
# 			
# 		return __wcc_directed(<DiGraph>G,nbunch,nbunch_,avg,ids,reverse)
# 	else:
# 		raise ZenException, 'Undirected graphs not yet supported'

cpdef __occ_directed(DiGraph G):
	cdef int num_nodes = 0
	cdef int idx
	cdef int x,i,j,k
	cdef int ni,nj,nk
	cdef int num_vees = 0
	cdef int num_tris = 0

	nodes = G.nodes_()
	num_nodes = len(nodes)

	for x in range(num_nodes):
		ni = nodes[x]

		num_vees += G.node_info[ni].outdegree * (G.node_info[ni].outdegree-1)
	
		# check each edge
		for i in range(G.node_info[ni].outdegree):
			nj = G.edge_info[G.node_info[ni].outelist[i]].tgt
			for j in range(G.node_info[nj].outdegree):
				nk = G.edge_info[G.node_info[nj].outelist[j]].tgt
				for k in range(G.node_info[nk].indegree):
					if G.edge_info[G.node_info[nk].inelist[k]].src == ni:
						num_tris += 1

	if num_vees == 0:
		return 0
	else:
		return <float> num_tris / <float> num_vees

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