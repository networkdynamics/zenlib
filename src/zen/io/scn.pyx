"""
The ``zen.io.scn`` module (available as ``zen.scn``) supports the reading and writing of graph data in a simple, concise network (SCN) format designed for quick parsing that resembles the :doc:`edgelist format <edgelist>`.

Description of the format
-------------------------

In general, each line of the file specifies either a node or an edge.  Networks are specified 
in two parts.  First nodes are specified, then edges are specified.  The sections are divided
by a line consisting of the character ``=``.

	* A node definition has the following format: ``<node name> <prop1> <prop2> ... <propN>``

	* An edge definition has the following format: ``<src> <dest> <prop1> <prop2> ... <propN>``

Other rules:
	* The first line in the file consists of two numbers: the number of properties nodes have and edges have
	* Node names and properties must not contain spaces and cannot start with an ``=`` sign
	* Lines that contain any text must start with that text (no leading whitespace)
	* The separator between entries on a line is *exactly* one space
	
Functions
---------

.. autofunction:: zen.io.scn.read

.. autofunction:: zen.io.scn.write
	
"""
from zen.digraph cimport DiGraph
from zen.graph cimport Graph
from zen.exceptions import *
import numpy

from cpython cimport bool

__all_ = ['read','write']

# include reading capabilities
cdef extern from "stdio.h" nogil:
	ctypedef struct FILE
		
	FILE* fopen(char* filename, char* mode)
	int fclose(FILE* stream)
	int fscanf(FILE* stream, char* format, ...)
	char* fgets(char* s, int n, FILE* stream)
	int fwrite(void* buffer, int size, int count, FILE* stream)
	bint feof(FILE* stream)
	bint isspace(int c)
	int ferror(FILE* stream)

def write(G,filename,**kwargs):
	"""
	Write the graph, ``G``, to the file named ``filename`` in the SCN format.
	
	Writing a graph in this format requires a bit more user-involvement than other formats 
	due to the strict interpretation of how properties are written out.
	
	When called, the number of node/edge properties that will be stored *for each* node/edge
	is specified (if not specified, it is assumed that zero properties will be written).  The
	actual properties that will be written for a node and edge are produced by ``node_data_fxn``
	and ``edge_data_fxn``, respectively.  These are functions which return a tuple containing string 
	properties of a specific node/edge, each property string containing no spaces. The tuple must have exactly the number of 
	entries as properties expected.  The node_data_fxn accepts three arguments: the node index,
	the node object, and the node data object.  The edge_data_fxn accepts four arguments: the edge
	index, the first (source) node, the second (target) node, and the edge data object.
	
	There is one exception for nodes and edges.  If a node/edge has no properties, 
	the node/edge_data_fxn may return None, in which case no properties will be associated
	with the node/edge.
	
	**KwArgs**:
	
		* ``num_node_props [=0]`` (int): the number of properties that will be written for each node.
		* ``num_edge_props [=0]`` (int): the number of properties that will be written for each edge.
		* ``use_node_indices [=False]`` (boolean): use the node index as its identifier rather than its object.
		* ``node_data_fxn [=None]``: the function that will be used to generate the node properties
		  that will be written to the file. This is a function which returns a tuple containing the string 
		  properties of the specific node provided in the parameters. Each property string must contain no spaces. 
		  Furthermore, the tuple must have exactly ``num_node_props`` entries (the number of properties expected).  
		  The ``node_data_fxn`` accepts three arguments: the node index, the node object, and the node data object.
		* ``edge_data_fxn [=None]``: the function that will be used to generate the edge properties
		  that will be written to the file. This is a function which returns a tuple containing the string 
		  properties of the specific edge provided in the parameters. Each property string must contain no spaces. 
		  Furthermore, the tuple must have exactly ``num_edge_props`` entries (the number of properties expected).  
		  The ``edge_data_fxn`` accepts four arguments: the edge index, the first (source) node index, the second 
		  (target) node index, and the edge data object.
	"""
	
	num_node_props = kwargs.pop('num_node_props',0)
	num_edge_props = kwargs.pop('num_edge_props',0)
	use_node_indices = kwargs.pop('use_node_indices',False)
	node_data_fxn = kwargs.pop('node_data_fxn',None)
	edge_data_fxn = kwargs.pop('edge_data_fxn',None)
	
	if len(kwargs) > 0:
		raise ZenException, 'Unknown keyword arguments: %s' % ', '.join(kwargs.keys())
		
	__inner_write(G,filename,num_node_props,num_edge_props,node_data_fxn,edge_data_fxn,use_node_indices)

cpdef __inner_write(G,filename,num_node_props,num_edge_props,node_data_fxn,edge_data_fxn,bool use_node_indices):
	if type(G) == DiGraph:
		write_digraph_scn(G,filename,num_node_props,num_edge_props,node_data_fxn,edge_data_fxn,use_node_indices)
	elif type(G) == Graph:
		write_graph_scn(G,filename,num_node_props,num_edge_props,node_data_fxn,edge_data_fxn,use_node_indices)
	else:
		raise ZenException, 'Graph of type %s not supported' % str(type(G))

cpdef write_graph_scn(G,filename,num_node_props,num_edge_props,node_data_fxn,edge_data_fxn,bool use_node_indices):
	fh = open(filename,'w')

	fh.write('%d %d\n' % (num_node_props,num_edge_props))

	# write nodes out
	if num_node_props > 0:
		if node_data_fxn == None:
			raise ZenException, 'node_data_fxn must be specified when nodes have properties'
			
		for nidx,n,data in G.nodes_iter_(obj=True,data=True):
			if use_node_indices:
				nstr = str(nidx)
			else:
				nstr = str(n)
			D = node_data_fxn(nidx,n,data)
			if D != None:
				fh.write(nstr + ' ' + ' '.join(D) + '\n')
	
	else: # write out the nodes
		if use_node_indices:
			for n_ in G.nodes_iter_():
				fh.write(str(n_) + '\n')
		else:
			for n in G.nodes_iter():
				fh.write(str(n) + '\n')

	# write edges
	fh.write('=\n')
	if num_edge_props > 0:
		if edge_data_fxn == None:
			raise ZenException, 'edge_data_fxn must be specified when edges have properties'
		
		for nidx in G.nodes_iter_():
			for eidx, data in G.edges_iter_(nidx, data=True):
				n1idx, n2idx = G.endpoints_(eidx)
				
				maxidx = n1idx if n1idx > n2idx else n2idx
				if nidx < maxidx: 
					continue
				
				# write out the edge (guarantees edges is written once only)
				n1,n2 = G.endpoints(eidx)
				D = edge_data_fxn(eidx,n1,n2,data)
				
				if use_node_indices:
					n1 = str(n1idx)
					n2 = str(n2idx)
				else:
					n1 = str(n1)
					n2 = str(n2)
					
				if D != None:
					fh.write(n1 + ' ' + n2 + ' ' + ' '.join(D) + '\n')
				else:
					fh.write(n1 + ' ' + n2 + '\n')
	else:
		for nidx in G.nodes_iter_():
			for eidx in G.edges_iter_(nidx):
				n1idx, n2idx = G.endpoints_(eidx)
				
				maxidx = n1idx if n1idx > n2idx else n2idx
				if nidx < maxidx: 
					continue

				n1,n2 = G.endpoints(eidx)
				
				if use_node_indices:
					n1 = str(n1idx)
					n2 = str(n2idx)
				else:
					n1 = str(n1)
					n2 = str(n2)
					
				fh.write(n1 + ' ' + n2 + '\n')
	# done
	fh.close()
	
cpdef write_digraph_scn(G,filename,num_node_props,num_edge_props,node_data_fxn,edge_data_fxn,bool use_node_indices):
	fh = open(filename,'w')

	fh.write('%d %d\n' % (num_node_props,num_edge_props))

	# write nodes out
	if num_node_props > 0:
		if node_data_fxn == None:
			raise ZenException, 'node_data_fxn must be specified when nodes have properties'
			
		for nidx,n,data in G.nodes_iter_(obj=True,data=True):
			if use_node_indices:
				nstr = str(nidx)
			else:
				nstr = str(n)
				
			D = node_data_fxn(nidx,n,data)
			if D != None:
				fh.write(nstr + ' ' + ' '.join(D) + '\n')

	else: # write out the nodes
		if use_node_indices:
			for n_ in G.nodes_iter_():
				fh.write(str(n_) + '\n')
		else:
			for n in G.nodes_iter():
				fh.write(str(n) + '\n')
	
	# write edges
	fh.write('=\n')
	if num_edge_props > 0:
		if edge_data_fxn == None:
			raise ZenException, 'edge_data_fxn must be specified when edges have properties'
		
		for eidx,data in G.edges_iter_(data=True):
			n1,n2 = G.endpoints(eidx)
			D = edge_data_fxn(eidx,n1,n2,data)
			
			if use_node_indices:
				n1,n2 = G.endpoints_(eidx)
				n1 = str(n1)
				n2 = str(n2)
			else:
				n1 = str(n1)
				n2 = str(n2)
				
			if D != None:
				fh.write(n1 + ' ' + n2 + ' ' + ' '.join(D) + '\n')
			else:
				fh.write(n1 + ' ' + n2 + '\n')
	else:
		if use_node_indices:
			for eidx in G.edges_iter_():
				n1_,n2_ = G.endpoints_(eidx)
				fh.write(str(n1_) + ' ' + str(n2_) + '\n')
		else:
			for n1,n2 in G.edges_iter():
				fh.write(str(n1) + ' ' + str(n2) + '\n')
		
	# done
	fh.close()
	
def read(filename,**kwargs):
	"""
	Read a network from the file with name ``filename`` specified in the SCN format.
	
	.. note: 
		The standard weighted argument is not supported as SCN does not currently specify a way of assigning weights to edges.
		
	**KwArgs**:
	
		* ``node_obj_fxn [=str]``: the function that converts the string node identifier read from the file
			into the node object
		* ``directed [=False]`` (boolean): whether the edges should be interpreted as directed.
		* ``ignore_duplicate_edges [=False]`` (boolean): ignore duplicate edges that may occur.  This incurs a performance
			hit since every edge must be checked before being inserted.
		* ``merge_graph [=None]`` (:py:class:`Graph` or :py:class:`DiGraph`): merge the edges read into the existing graph object provided. In this case,
			the ``merge_graph`` is returned (rather than a new graph object).
		* ``weighted [=False]`` (boolean): a third column of numbers will be expected in the file and will be interpreted 
			as edge weights.
		* ``max_line_len [=500]`` (int): this method is implemented in C for speed.  As part of the implementation, a fixed size buffer is used to minimize time
			spent parsing each line.  By default, the buffer size is 500 characters.  This argument allows a different max size
			buffer to be set.
			

	**Returns**:
		:py:class:`Graph` or :py:class:`DiGraph`. The graph constructed from the graph data read in SCN format.  If ``directed = True``, then a :py:class:`DiGraph` is built,
		otherwise a :py:class:`Graph` is built.  In the graph object that is returned, node and edge properties are stored in a numpy array that is assigned to be the data
		object of the respective node.
	"""
	
	node_obj_fxn = kwargs.pop('node_obj_fxn',None)
	directed = kwargs.pop('directed',None)
	ignore_duplicate_edges = kwargs.pop('ignore_duplicate_edges',False)
	merge_graph = kwargs.pop('merge_graph',None)
	max_line_len = kwargs.pop('max_line_len',500)
	
	if len(kwargs) > 0:
		raise ZenException, 'Unknown keyword arguments: %s' % ', '.join(kwargs.keys())
	
	if directed is True or (directed is None and type(merge_graph) == DiGraph):
		return parse_directed_scn(filename,max_line_len,node_obj_fxn,ignore_duplicate_edges,merge_graph)
	else:
		return parse_undirected_scn(filename,max_line_len,node_obj_fxn,ignore_duplicate_edges,merge_graph)
			
cdef parse_directed_scn(char* filename,int max_line_len,node_obj_fxn,bool ignore_duplicate_edges,merge_graph):
	
	cdef DiGraph G = None
	
	if merge_graph is None:
		G = DiGraph()
	elif type(merge_graph) == DiGraph:
		G = merge_graph
	else:
		raise ZenException, 'The merge_graph must be a DiGraph if directed information will be read from scn source'

	# open the file
	cdef FILE* fh
	
	fh = fopen(filename,'r')

	if fh == NULL:
		raise ZenException, 'Unable to open file %s' % filename
		
	# make the string buffer
	last_str_buffer = '0'*max_line_len
	str_buffer = '0'*max_line_len
	
	cdef char* buffer = str_buffer
	
	cdef int start1, start2, end1, end2
	cdef int line_no = 0
	cdef int buf_len
	cdef int i
	cdef bool reading_nodes = True
	cdef int nid1, nid2
	
	# read the number of properties
	while (1):
		fgets(buffer,max_line_len,fh)
		if buffer[0] != '#': 
			break # ignore comments until line with number of properties

	buf_len = len(buffer)
	
	start1 = 0
	for i in range(start1+1,buf_len):
		if isspace(<int>buffer[i]):
			break
	end1 = i
	start2 = end1+1
	for i in range(start2+1,buf_len):
		if isspace(<int>buffer[i]):
			break
	end2 = i
	
	cdef int num_node_props = int(buffer[start1:end1])
	cdef int num_edge_props = int(buffer[start2:end2])
	
	while not feof(fh):
		line_no += 1
		
		fgets(buffer,max_line_len,fh)		
		buf_len = len(buffer)

		if buffer[0] == '#':
			continue # ignore comments in node or edge section
		
		# check the success
		if buffer[buf_len-1] != '\n' and not feof(fh):
			raise ZenException, 'Line %d exceeded maximum line length (%d)' % (line_no,max_line_len)
		
		# make sure we aren't reprocessing a line of input
		if buffer[buf_len-1] == '\n' and feof(fh):
			break
		
		# skip whitespace
		if isspace(<int>buffer[0]):
			continue
		
		# check for the node to edge switch
		if buffer[0] == '=':
			reading_nodes = False
			continue
		
		start1 = 0
		
		for i in range(start1+1,buf_len):
			if isspace(<int>buffer[i]):
				break
		end1 = i
	
		if reading_nodes:
			if node_obj_fxn is None:
				name = buffer[start1:end1]
			else:
				name = node_obj_fxn(buffer[start1:end1])
			props = None
			if num_node_props > 0:
				props = process_properties(buffer,buf_len,end1+1,line_no,num_node_props)
			
			G.add_node(name,props)
		else:
			start2 = end1+1
			
			for i in range(start2+1,buf_len):
				if isspace(<int>buffer[i]):
					break
			end2 = i
			
			if end2 == buf_len-1 and not isspace(<int>buffer[end2]):
				end2 += 1
			
			if node_obj_fxn is None:
				name1 = buffer[start1:end1]
				name2 = buffer[start2:end2]
			else:
				name1 = node_obj_fxn(buffer[start1:end1])
				name2 = node_obj_fxn(buffer[start2:end2])
				
			props = None
			if num_edge_props > 0 and (end2+1 < buf_len and not isspace(<int>buffer[end2+1])):
				props = process_properties(buffer,buf_len,end2+1,line_no,num_edge_props)
				
			nid1 = -1
			nid2 = -1
			
			# handle any nodes that didn't have a definition line
			if name1 in G.node_idx_lookup:
				nid1 = G.node_idx_lookup[name1]
			else:
				nid1 = G.add_node(name1)
				
			if name2 in G.node_idx_lookup:
				nid2 = G.node_idx_lookup[name2]
			else:
				nid2 = G.add_node(name2)
			
			if ignore_duplicate_edges and G.has_edge_(nid1,nid2):
				continue
				
			G.add_edge_(nid1,nid2,props)
	
	fclose(fh)
	
	return G

cdef parse_undirected_scn(char* filename,int max_line_len,node_obj_fxn,bool ignore_duplicate_edges,merge_graph):

	cdef Graph G = None

	if merge_graph is None:
		G = Graph()
	elif type(merge_graph) == Graph:
		G = merge_graph
	else:
		raise ZenException, 'The merge_graph must be a Graph if undirected information will be read from scn source'

	# open the file
	cdef FILE* fh

	fh = fopen(filename,'r')

	if fh == NULL:
		raise ZenException, 'Unable to open file %s' % filename

	# make the string buffer
	last_str_buffer = '0'*max_line_len
	str_buffer = '0'*max_line_len

	cdef char* buffer = str_buffer

	cdef int start1, start2, end1, end2
	cdef int line_no = 0
	cdef int buf_len
	cdef int i
	cdef bool reading_nodes = True
	cdef int nid1, nid2

	# read the number of properties
	while (1):
		fgets(buffer,max_line_len,fh)
		if buffer[0] != '#': 
			break # ignore comments until line with number of properties

	buf_len = len(buffer)

	start1 = 0
	for i in range(start1+1,buf_len):
		if isspace(<int>buffer[i]):
			break
	end1 = i
	start2 = end1+1
	for i in range(start2+1,buf_len):
		if isspace(<int>buffer[i]):
			break
	end2 = i

	cdef int num_node_props = int(buffer[start1:end1])
	cdef int num_edge_props = int(buffer[start2:end2])

	while not feof(fh):
		line_no += 1

		fgets(buffer,max_line_len,fh)		
		buf_len = len(buffer)

		if buffer[0] == '#':
			continue # ignore comments in node or edge section

		# check the success
		if buffer[buf_len-1] != '\n' and not feof(fh):
			raise ZenException, 'Line %d exceeded maximum line length (%d)' % (line_no,max_line_len)

		# make sure we aren't reprocessing a line of input
		if buffer[buf_len-1] == '\n' and feof(fh):
			break

		# skip whitespace
		if isspace(<int>buffer[0]):
			continue

		# check for the node to edge switch
		if buffer[0] == '=':
			reading_nodes = False
			continue

		start1 = 0

		for i in range(start1+1,buf_len):
			if isspace(<int>buffer[i]):
				break
		end1 = i

		if reading_nodes:
			if node_obj_fxn is None:
				name = buffer[start1:end1]
			else:
				name = node_obj_fxn(buffer[start1:end1])
			props = None
			if num_node_props > 0:
				props = process_properties(buffer,buf_len,end1+1,line_no,num_node_props)

			G.add_node(name,props)
		else:
			start2 = end1+1

			for i in range(start2+1,buf_len):
				if isspace(<int>buffer[i]):
					break
			end2 = i

			if end2 == buf_len-1 and not isspace(<int>buffer[end2]):
				end2 += 1

			if node_obj_fxn is None:
				name1 = buffer[start1:end1]
				name2 = buffer[start2:end2]
			else:
				name1 = node_obj_fxn(buffer[start1:end1])
				name2 = node_obj_fxn(buffer[start2:end2])

			props = None
			if num_edge_props > 0 and (end2+1 < buf_len and not isspace(<int>buffer[end2+1])):
				props = process_properties(buffer,buf_len,end2+1,line_no,num_edge_props)

			nid1 = -1
			nid2 = -1

			# handle any nodes that didn't have a definition line
			if name1 in G.node_idx_lookup:
				nid1 = G.node_idx_lookup[name1]
			else:
				nid1 = G.add_node(name1)

			if name2 in G.node_idx_lookup:
				nid2 = G.node_idx_lookup[name2]
			else:
				nid2 = G.add_node(name2)

			if ignore_duplicate_edges and G.has_edge_(nid1,nid2):
				continue

			G.add_edge_(nid1,nid2,props)

	fclose(fh)

	return G
			
cdef process_properties(char* buffer, int buf_len, int i, int line_no, int num_props):
	"""
	Upon entry, buffer[idx] = '['.  In other words, this function should only
	be called when a property block *does* exist.
	"""
	cdef int p, sprop, eprop
	
	props = numpy.empty( num_props, dtype=numpy.object_)
	
	for p in range(num_props):
		sprop = i
		for i in range(sprop+1,buf_len):
			if isspace(<int>buffer[i]):
				break
		eprop = i
		if eprop == buf_len-1 and not isspace(<int>buffer[eprop]):
			eprop += 1
			
		prop = buffer[sprop:eprop]
		props[p] = prop
		
		# move past the space
		i += 1
		
	return props

