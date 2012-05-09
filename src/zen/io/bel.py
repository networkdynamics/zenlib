"""
The ``zen.io.bel`` module (available as ``zen.bel``) supports the reading and writing of edge lists in binary format.  The binary format was designed
by Derek Ruths for use with Zen.  In a binary format, each node is referenced by an index number (node/edge
labels and properties are not supported).  The complete format is:

``<F = Format Version #><N = # of nodes> <M = # of edges> <u1><v1><u2><v2>...<um><vm>`` where:
	
	* ``F`` is a 8 bit number indicating the format version used, 
	* ``N`` is a 32 bit number indicating the max index number (nodes are indexed from zero to ``N``),
	* ``E`` is a 64 bit number indicating the number of edges stored, and
	* ``ui`` and ``vi`` are ``ceil(log2(N+1))`` bit numbers storing the index numbers of the left and right nodes for edge ``i``.
	
The current version of the binary edge list (.bel) format is 1.0.

Reading
-------

Graphs can be read either from file-like objects using :py:func:`zen.io.bel.read` or from strings using :py:func:`zen.io.bel.read_str`.
Both functions accept the same set of arguments:

	* ``read(fh,<keyword arguments>)``
	* ``read_str(sbuffer,<keyword arguments>)``

``fh/sbuffer`` specifies the object from which the graph will be read.

Supported keywords include the following:

	* ``node_obj_fxn [= str]``: unlike the default definition, this function accepts integers and returns the node object
	
	* ``directed [= False]`` (boolean): indicates whether the data is read as directed
	
	* ``check_for_duplicates [= False]`` (boolean): applies only when loading an undirected graph. If True, then a check will be made to ensure that
	  no duplicate edges are attempted to be added (in case the underlying graph was originally directed).  Checking incurs a small
	  performance cost due to the check.

.. autofunction:: zen.io.bel.read(fh[,node_obj_fxn=str,directed=False])

.. autofunction:: zen.io.bel.read_str(fh[,node_obj_fxn=str,directed=False])

Writing
-------

Graphs can be written either to file-like objects using :py:func:`zen.io.bel.write` or to strings using :py:func:`write_str`.
Both functions accept the similar sets of arguments:

	* ``write(G,fh)``
	* ``write_str(G)``

``G`` specifies the graph to write. ``fh`` is a file-like object.  ``write_str(...)`` returns the string
representation of the ``G``, so no argument equivalent of ``fh`` exists).

``G`` must be compact (an exception will be raised if the graph is not compact).  The graph is stored in 
the binary edge list format and node indexes are used to index the edge list when writing.

.. autofunction:: zen.io.bel.write(G,fh)

.. autofunction:: zen.io.bel.write_str(G)

Citations
---------

Techniques similar to the ones used here for network compression have been explored previously in publications
such as 

.. note::
	B. Dengiz et al. Local search genetic algorithm for optimal design of reliable networks.
	IEEE Transactions on Evolutionary Computation, 1(3):179-188, 1997.
	
The general idea is drawn from variable length integer encoding which has been used in ZIP compression, for example.
"""

from zen.constants import *
from zen.exceptions import *
from zen.graph import Graph
from zen.digraph import DiGraph

from zen.util.bitvector import BitVector
import math
import array
import types

__all__ = ['write','write_str','read','read_str']

VERSION_LEN = 8
NUM_INDEX_LEN = 32
NUM_EDGES_LEN = 64

def read_str(sbuffer, **kwargs):
	"""
	Read graph data from the ascii string in the binary edge list format.
	
	**Args**:
		``sbuffer`` is the string from which the network data will be read.
	
	**KwArgs**:
		* ``node_obj_fxn [= str]``: unlike the default definition, this function accepts integers and returns the node object
	
		* ``directed [= False]`` (boolean): indicates whether the data is read as directed
	
		* ``ignore_duplicate_edges [= False]`` (boolean): applies only when loading an undirected graph. If True, then a check will be made to ensure that
	  	  no duplicate edges are attempted to be added (in case the underlying graph was originally directed).  Checking incurs a small
	  	  performance cost due to the check.
	
	**Returns**:
		:py:class:`zen.Graph` or :py:class:`zen.DiGraph`.  The graph read from the input string.  The ``directed`` parameter decides
		whether a directed or undirected graph is constructed.
	"""
	
	# handle the keyword arguments
	node_obj_fxn = kwargs.pop('node_obj_fxn',str)
	directed = kwargs.pop('directed',False)
	check_for_duplicates = kwargs.pop('ignore_duplicate_edges',False)
	
	if len(kwargs) > 0:
		raise ZenException, 'Unknown keyword arguments: %s' % ', '.join(kwargs.keys())
	
	if check_for_duplicates and directed:
		raise ZenException, 'ignore_duplicate_edges can only be set when directed = False'
		
	# build the graph
	G = None
	if directed == True:
		G = DiGraph()
	elif directed == False:
		G = Graph()
	else:
		raise ZenException, 'directed must be either True or False.'
	
	#####
	# convert the string into a bitvector
	bv = BitVector(size = len(sbuffer) * 8)
	offset = 0
	for c in sbuffer:
		v = ord(c)
		dec2bv(v,bv,offset,8)
		offset += 8
		
	#####
	# read the header
	offset = 0
	
	# read the version
	version = bv2dec(bv,offset,VERSION_LEN)
	offset += VERSION_LEN
	
	if version != 1:
		raise Exception, 'Invalid file format or version number'
	
	# read the num of indexes
	last_idx = bv2dec(bv,offset,NUM_INDEX_LEN)
	idx_size = int(math.ceil(math.log(last_idx+1,2)))
	
	offset += NUM_INDEX_LEN
	idx2node = [None] * (last_idx + 1)
	
	# build all nodes right now
	if node_obj_fxn is not None:
		for x in xrange(last_idx+1):
			n = node_obj_fxn(x)
			G.add_node(n)
	else:
		G.add_nodes(last_idx+1)
	
	# read the number of edges
	num_edges = bv2dec(bv,offset,NUM_EDGES_LEN)
	offset += NUM_EDGES_LEN
	
	#####
	# Read the content: every edge
	if directed:
		for ei in xrange(num_edges):
			idx1 = bv2dec(bv,offset,idx_size)
			offset += idx_size
			idx2 = bv2dec(bv,offset,idx_size)
			offset += idx_size
				
			G.add_edge_(idx1,idx2)
	else:
		for ei in xrange(num_edges):
			idx1 = bv2dec(bv,offset,idx_size)
			offset += idx_size
			idx2 = bv2dec(bv,offset,idx_size)
			offset += idx_size
			
			if check_for_duplicates and G.has_edge_(idx1,idx2):
				continue
				
			G.add_edge_(idx1,idx2)		
	
	# done!	
	return G
	
def read(fh, **kwargs):
	"""
	Read graph data from the file-like object in the edge list format.
	
	
	
	**Args**:
		``fh`` (file-handle or string): if `fh` is a string, then it is interpreted as a filename that the network will be read from.
		If ``fh`` is a file-handle, then the network data is read from the data returned from the file stream.
	
	**KwArgs**:
		* ``node_obj_fxn [= str]``: unlike the default definition, this function accepts integers and returns the node object

		* ``directed [= False]`` (boolean): indicates whether the data is read as directed

		* ``ignore_duplicate_edges [= False]`` (boolean): applies only when loading an undirected graph. If True, then a check will be made to ensure that
  	  	  no duplicate edges are attempted to be added (in case the underlying graph was originally directed).  Checking incurs a small
		  performance cost due to the check.
	
	**Returns**:
		:py:class:`zen.Graph` or :py:class:`zen.DiGraph`.  The graph read from the input file.  The ``directed`` parameter decides
		whether a directed or undirected graph is constructed.
	"""
	
	close_fh = False
	if type(fh) == types.StringType:
		close_fh = True
		fh = open(fh,'r')
	
	data = fh.read()
	G = read_str(data, **kwargs)
	
	if close_fh:
		fh.close()
	
	return G

def construct_bitvector(G, max_index):
	"""
	Create a bitvector for the graph's binary edge list representation.  Initialize its header
	region.  Return values:
	
		- bitvector
		- length of header
		- number of bits to be used in representing the index
	"""
	header_len = VERSION_LEN + NUM_INDEX_LEN + NUM_EDGES_LEN
	idx_size = int(math.ceil(math.log(max_index+1,2)))
	data_len = 2 * idx_size * G.size()
	total_size = header_len + data_len
	
	bv = BitVector(size=total_size)
	offset = 0
	
	#####
	# Write the header
	
	# write the version number
	dec2bv(1,bv,offset,VERSION_LEN)
	offset += VERSION_LEN
	
	# write the max node index fxn
	dec2bv(max_index,bv,offset,NUM_INDEX_LEN)
	offset += NUM_INDEX_LEN
	
	# write the num of edges
	dec2bv(G.size(),bv,offset,NUM_EDGES_LEN)
	offset += NUM_EDGES_LEN
	
	return bv, header_len, idx_size
	
def store_bitvector_strict_order(G, max_index, node_index_fxn):
	"""
	This internal function writes a graph to the binary edge list format in a bitvector.
	Unlike store_bitvector(...), this method writes the edges in a very specific order:
	
		For any two edges e1 = (u,v) and e2 = (x,y), e1 is written before e2 iff idx(u) < idx(x).
		If idx(u) == idx(x), then e1 is written before e2 iff idx(v) < idx(y).
		
	In undirected graphs, edge endpoints (u,v) are always ordered such that idx(u) < idx(v).
	"""
	# build the bitvector and header
	bv, offset, idx_size = construct_bitvector(G, max_index)
	
	# make a node index lookup
	node2index_lookup = {}
	for n in G.nodes_iter():
		idx = node_index_fxn(n)
		node2index_lookup[n] = (idx,dec2bin(idx,idx_size))
		
	#####
	# Write the data section
	for u in G.nodes_iter():
		uidx = node2index_lookup[u]
		edges = []
		for e in G.edges_iter(u):
			v = G.endpoint(e,u)
			vidx = node2index_lookup[v]
			
			# only keep edges for which u is the smaller node
			if uidx[0] < vidx[0]:
				edges.append(vidx)
				
		# order the edges by the second endpoint
		edges.sort(cmp = lambda x,y: cmp(x[0],y[0]))
		
		# write them...
		a1 = uidx[1]
		for e in edges:
			for i in xrange(idx_size):
				bv[offset + i] = a1[i] == '1'
			offset += idx_size
			
			a2 = vidx[1]
			for i in xrange(idx_size):
				bv[offset + i] = a2[i] == '1'
			offset += idx_size
	
	####
	# done!
	return bv

def store_bitvector(G):
	"""
	This is an internal function that writes a graph to the binary edge list format in a bitvector.
	"""
	max_index = len(G) - 1
	
	# build the bitvector and header
	bv, offset, idx_size = construct_bitvector(G, max_index)
	
	# make a node index lookup
	node2index_lookup = {}
	for idx in G.nodes_iter_():
		node2index_lookup[idx] = dec2bin(idx,idx_size)
	
	#####
	# Write the data section
	for e in G.edges_iter_():
		x,y = G.endpoints(e)
		
		a1 = node2index_lookup[x]
		a2 = node2index_lookup[y]
	
		for i in xrange(idx_size):
			bv[offset + i] = a1[i] == '1'
		offset += idx_size
		
		for i in xrange(idx_size):
			bv[offset + i] = a2[i] == '1'
		offset += idx_size
	
	return bv
	
def write_str(G):
	"""
	Write the graph, ``G``, to a binary edge list representation and return this in an ascii string.
	
	.. note::
		The graph must be compact in order to be written.  If it is not compact, an exception will be raised. See :py:meth:`zen.Graph.compact`.
	
	**Raises**:
		:py:exc:`zen.ZenException`: if the graph is not compact.
	"""
	
	if not G.is_compact():
		raise ZenException, 'Graph G must be compact'
		
	bv = None
	# if strict_order:
	# 	bv = store_bitvector_strict_order(G)
	# else:
	# 	bv = store_bitvector(G)
	bv = store_bitvector(G)
	
	# walk through the bitvector 8 bit chunks, convert each to a character
	num_chars = int(math.ceil(float(len(bv)) / 8.))
	result = array.array('c',['0'] * num_chars)
	
	for i in xrange(num_chars):
		start = i * 8
		end = (i+1) * 8
		true_end = end
		n = 0
		if end > len(bv):
			end = len(bv)

		for j in xrange(start,end):
			n <<= 1
			if bv[j]:
				n |= 1

		if end < true_end:
			n <<= (true_end - end)
			
		result[i] = chr(n)
		
	return result.tostring()
	
def write(G, fh):
	"""
	Write the graph, ``G``, to the file-like object fh in the binary edge list format.  If `fh` is a string, then it is 
	interpreted as the name of the file to which the graph will be written.
	
	.. note::
		The graph must be compact in order to be written.  If it is not compact, an exception will be raised. See :py:meth:`zen.Graph.compact`.
	
	**Raises**:
		:py:exc:`zen.ZenException`: if the graph is not compact.
	"""
	close_fh = False
	if type(fh) == types.StringType:
		close_fh = True
		fh = open(fh,'w')
	
	# convert the network to a string representation of the binary edge list format
	str_rep = write_str(G)
	
	# write the string to the file
	fh.write(str_rep)
	
	if close_fh:
		fh.close()
	
	# done!
	return

def dec2bv(i,bv,offset,max_len):
	"""
	Store a decimal integer to the specified bitvector at the specified position
	"""
	num_bits = math.ceil(math.log(i+1,2))
	
	if num_bits > max_len:
		raise Exception, 'Too many bits being stored.'
	
	pos = max_len + offset - 1
	while i > 0:
		bv[pos] = i % 2 == 1
		i = i >> 1
		pos -= 1

	return

def dec2bin(i,max_len):
	"""
	Store a decimal integer to an array of characters '0' = 0, '1' = 1
	"""
	num_bits = math.ceil(math.log(i+1,2))
	
	if num_bits > max_len:
		raise Exception, 'Too many bits being stored.'
	
	A = array.array('c',['0'] * max_len)
	pos = max_len - 1
	while i > 0:
		if i % 2 == 1:
			A[pos] = '1'
		i = i >> 1
		pos -= 1

	return A

def bv2dec(bv,offset,length):
	n = 0
	
	for pos in xrange(offset,offset+length):
		n <<= 1
		if bv[pos] == True:
			n |= 1

	return n

#####
# For testing		
def main():
	
	G = Graph()
	G.E.add(1,2)
	G.E.add(2,3)
	G.E.add(3,4)
	idx_lookup = dict([(n,i) for n, i in zip(G.V,xrange(len(G.V)))])
	#bv = store_bitvector(G, len(idx_lookup)-1, lambda x: idx_lookup[x])
	s = store_str(G, strict_order = True)
	G2 = read_str(s)
	print len(G2.V), len(G2.E), G2.topology()

if __name__ == '__main__':
	main()

