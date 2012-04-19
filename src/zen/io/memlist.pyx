"""
This module implements a memory-mapped edge list in which edges are specified by 
their node index pairs.  This format can be read very quickly.  These files, however,
must be created very carefully.

== The Memory-mapped Edge List format ==
The first line of the file indicates the number of nodes in the network - this will be taken
to be an upper-bound on the node index values appearing in the edge list.

All remaining lines consist of node index pairs that specify edges - one edge per line.

Other rules:
	- Lines that contain any text must start with that text (no leading whitespace)
	- The separator between entries is any whitespace
	- Any line beginning with a '#' character is treated as a comment.
	- In order to write out a graph in this format, it must be compacted.  This means that there cannot be any unallocated
	  nodes in the node array prior to the last allocated node (e.g., the node array can't be fragmented).
	
"""
from zen.digraph cimport DiGraph
from zen.graph cimport Graph
from zen.exceptions import *

from cpython cimport bool

__all__ = ['read','write']

# include reading capabilities
cdef extern from "stdio.h" nogil:
	ctypedef struct FILE
	
	FILE* fopen(char* filename, char* mode)
	int fclose(FILE* stream)
	int fscanf(FILE* stream, char* format, ...)
	char* fgets(char* s, int n, FILE* stream)
	bint feof(FILE* stream)
	bint isspace(int c)
	int ferror(FILE* stream)
	
def write(G,filename):
	"""
	Write the graph in memlist format to the file specified. Node values
	will be node indices.  This assumes that the graph has been compacted.
	If the graph is not compact, then a ZenException will be raised.
	
	Note that the standard use_node_indices keyword is not supported here because memlist
	always uses the node indices (rather than node objects).
	"""
	if not G.is_compact():
		raise ZenException, 'Graph is not compact'
		
	__inner_write(G,filename)
	
cpdef __inner_write(G,filename):
	fh = open(filename,'w')
		
	fh.write('%d\n' % (G.max_node_idx+1))
		
	for eid in G.edges_iter_():
		fh.write('%d %d\n' % G.endpoints_(eid))
		
	fh.close()
	
def read(char* filename,**kwargs):
	"""
	Read in a network from a file in a memory mapped edge list format.
	
	Keyword arguments suppported are:
	
		- node_obj_fxn [=None]: the function that accepts an integer node id read from the memlist file
			and returns the node object.  Since memlist is designed to handle very large networks, by default
			no node objects are added to the graph.
		- directed [=False]: whether the edges should be interpreted as directed (if so, a DiGraph object
			will be returned)
		- ignore_duplicate_edges [=False]: ignore duplicate edges that may occur.  This incurs a performance
			hit since every edge must be checked before being inserted.
		- weighted [=False]: a third column of numbers will be expected in the file and will be interpreted 
			as edge weights.
	"""
	
	directed = kwargs.pop('directed',None)
	ignore_duplicate_edges = kwargs.pop('ignore_duplicate_edges',False)
	weighted = kwargs.pop('weighted',False)
	node_obj_fxn = kwargs.pop('node_obj_fxn',None)
	
	if len(kwargs) > 0:
		raise ZenException, 'Unknown keyword arguments: %s' % ', '.join(kwargs.keys())
	
	return __inner_read(filename,directed,ignore_duplicate_edges,weighted,node_obj_fxn)
	
cpdef __inner_read(char* filename,bool directed,bool ignore_duplicates,bool weighted,node_obj_fxn):
	
	if directed is True:
		G = DiGraph()
	else:
		G = Graph()
	
	cdef FILE* fh
	cdef int MAX_LINE_LEN = 100
	
	# make the string buffer
	str_buffer = '0'*MAX_LINE_LEN
	
	cdef char* buffer = str_buffer

	# open the file
	fh = fopen(filename,'r')
	
	if fh == NULL:
		raise Exception, 'Unable to open file %s' % filename
	
	# read all the lines	
	cdef int start1, start2, end1, end2, start3, end3
	cdef int nidx1,nidx2
	cdef double w
	cdef int line_no = 0
	cdef int buf_len
	cdef int i
	
	# read the number of nodes
	cdef int num_nodes = -1
	while num_nodes < 0:
		line_no += 1
		result = fgets(buffer,MAX_LINE_LEN,fh)
		
		# if the result is NULL, we've hit the end of the file
		if result is NULL:
			# get out of the reading loop
			break
			
		if buffer[0] == '#': 
			continue # ignore comments until line with number of properties

		buf_len = len(buffer)
		
		# check the success
		if not feof(fh) and buffer[buf_len-1] != '\n':
			raise ZenException, 'Line %d exceeded maximum line length (%d)' % (line_no,MAX_LINE_LEN)
			
		# get the number of nodes
		num_nodes = int(str(buffer).strip())
		
	if num_nodes == -1:
		raise ZenException, 'The number of nodes was not specified'
		
	G.add_nodes(num_nodes,node_obj_fxn)
	
	while not feof(fh):
		line_no += 1
		
		start1 = 0; start2 = 0; end1 = 0; end2 = 0; start3 = 0; end3 = 0;
		
		result = fgets(buffer,MAX_LINE_LEN,fh)
		
		# if the result is NULL, we've hit the end of the file
		if result is NULL:
			# get out of the reading loop
			break
			
		if buffer[0] == '#': 
			continue # ignore comments until line with number of properties

		buf_len = len(buffer)
		
		# check the success
		if not feof(fh) and buffer[buf_len-1] != '\n':
			raise ZenException, 'Line %d exceeded maximum line length (%d)' % (line_no,MAX_LINE_LEN)
			
		# find the first element
		for i in range(buf_len):
			if not isspace(<int>buffer[i]):
				break
		start1 = i
		for i in range(start1+1,buf_len):
			if isspace(<int>buffer[i]):
				break
		end1 = i
		
		for i in range(end1+1,buf_len):
			if not isspace(<int>buffer[i]):
				break
		start2 = i
		for i in range(start2+1,buf_len):
			if isspace(<int>buffer[i]):
				break
		end2 = i
		
		if weighted:
			for i in range(end2+1,buf_len):
				if not isspace(<int>buffer[i]):
					break
			start3 = i
			for i in range(start3+1,buf_len):
				if isspace(<int>buffer[i]):
					break
			end3 = i
			
			if end3 == buf_len-1 and not isspace(<int>buffer[end3]):
				end3 += 1
				
			w = float(buffer[start3:end3])
		else:		
			if end2 == buf_len-1 and not isspace(<int>buffer[end2]):
				end2 += 1
		
		if start1 >= end1 or start2 >= end2 or (weighted and start3 >= end3):
			raise Exception, 'Line %d was incorrectly formatted: %s' % (line_no,buffer)
				
		nidx1 = int(buffer[start1:end1])
		nidx2 = int(buffer[start2:end2])
		
		if nidx1 >= num_nodes or nidx2 >= num_nodes:
			raise ZenException, 'Line %d: edge (%d,%d) referenced a node with index larger than max number of nodes (%d)' % (line_no,nidx1,nidx2,num_nodes)
		
		if ignore_duplicates and G.has_edge_(nidx1,nidx2):
			continue
			
		G.add_edge_(nidx1,nidx2,weight=w) if weighted else G.add_edge_(nidx1,nidx2)
				
	fclose(fh)
	
	return G
