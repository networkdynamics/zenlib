#cython: embedsignature=True

"""
This module provides code for reading and writing networks in the restricted dot (rdot) format.

The restricted DOT format contains a subset of features of the entire file format.  The restrictions
are primarily intended to accelerate the speed of reading and writing networks to the format while
retaining most of its expressive power.

== rDOT Format Description ==
	- In the rDOT format, a graph can contain nodes and edges (but not subgraphs, etc...).  Both nodes and edges
		can have properties with string values which are always encapsulated in quotes.
		
	- Node names cannot contain spaces, '-' (dash), or '>'.  They also cannot be encapsulated in quotes.

	- Graphs are either all directed (in which case the graph should have type "digraph") or undirected ("graph").
		Mixed graphs are not supported.

	- Lines can be at most 500 characters long and a node or edge definition must fit on a single line.

	- Comments are not supported.
	
	- Each file must contain exactly one graph.
	
	- All node declarations must occur first, followed by the edge declarations.
	
	- No graph properties are allowed.
"""

from zen.digraph cimport DiGraph
from zen.exceptions import *

from cpython cimport bool

__all_ = ['read']

cdef int MAX_LINE_LEN = 500

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
	
def read(filename,**kwargs):
	"""
	Read a network from the file specified in restricted DOT format.
	
	Keyword arguments suppported are:
	
		- node_obj_fxn [=str]: the function that converts the string node identifier read from the file
			into the node object
		- merge_graph [=None]: merge the edges read into the existing graph object provided. In this case,
			the merge_graph is returned (rather than a new graph object).
			
	The standard keywords directed and ignore_duplicate_edges are not supported because the rdot format explicitly
	indicates the directionality of graphs.  Since directed graphs cannot be read in undirected form, ignore_duplicate_edges
	is not relevant.  The weights keyword is not supported because all attributes of nodes and edges are loaded,
	thus weights will be loaded if appropriate.
	"""
	node_obj_fxn = kwargs.pop('node_obj_fxn',str)
	merge_graph = kwargs.pop('merge_graph',None)
	
	return __inner_read(filename,node_obj_fxn,merge_graph)
	
cpdef __inner_read(filename,node_obj_fxn,merge_graph):
	cdef FILE* fh
	
	# make the string buffer
	str_buffer = '0'*MAX_LINE_LEN

	cdef char* buffer = str_buffer

	# open the file
	fh = fopen(filename,'r')

	if fh == NULL:
		raise ZenException, 'Unable to open file %s' % filename

	######
	# read the header to decide whether we're reading directed or undirected	
	cdef int buf_len
	cdef int i

	# find the header
	while not feof(fh):
		fgets(buffer,MAX_LINE_LEN,fh)
		buf_len = len(buffer)
		
		# find the first non-space character
		for i in range(buf_len):
			if not isspace(<int>i):
				break
				
		if i == buf_len or isspace(<int>buffer[i]):
			continue
			
		if buffer[i] == 'd': # digraph
			G = parse_directed_rdot(fh,node_obj_fxn,merge_graph)
			fclose(fh)
			return G
		elif buffer[i] == 'g': # graph
			raise ZenException, 'Undirected graphs are currently not supported'
		else:
			raise ZenException, 'Expected to find either "graph" or "digraph"'
			
cdef parse_directed_rdot(FILE* fh,node_obj_fxn,merge_graph):
	
	cdef DiGraph G = None
	
	if merge_graph is None:
		G = DiGraph()
	elif type(merge_graph) == DiGraph:
		G = merge_graph
	else:
		raise ZenException, 'merge_graph must be a DiGraph to incorporate directed rdot information'	

	# make the string buffer
	str_buffer = '0'*MAX_LINE_LEN

	cdef char* buffer = str_buffer

	cdef int start1, start2, end1, end2
	cdef int line_no = 0
	cdef int buf_len
	cdef int i
	cdef bool reading_nodes = True
	
	while not feof(fh):
		line_no += 1
		
		fgets(buffer,MAX_LINE_LEN,fh)
		if feof(fh):
			line_no -= 1
			break
		
		buf_len = len(buffer)
		
		# check the success
		if buffer[buf_len-1] != '\n':
			raise Exception, 'Line %d exceeded maximum line length (%d)' % (line_no,MAX_LINE_LEN)
		
		# find the first element
		for i in range(buf_len):
			if not isspace(<int>buffer[i]):
				break
			
		if i == (buf_len - 1):
			# if it's an empty line, skip it.
			continue
		start1 = i
		
		# if we've reached the end
		if buffer[start1] == '}':
			break
		
		for i in range(start1+1,buf_len):
			if isspace(<int>buffer[i]) or buffer[i] == ';' or buffer[i] == '[':
				break
		end1 = i
	
		# find the beginning of the second entry
		for i in range(end1,buf_len):
			if not isspace(<int>buffer[i]):
				break
		start2 = i
		
		if reading_nodes:
			# check to see if we've found an edge
			if not buffer[start2] == '[' and not buffer[start2] == ';':
				reading_nodes = False
			else:
				name = buffer[start1:end1]
				props = None
				if buffer[start2] == '[':
					props = process_properties(buffer,buf_len,start2,line_no)
				G.add_node(name,props)
				
		# this is not an elif because it's possible that while reading a "node", we discover it's an edge
		# and need to drop into this case.
		if not reading_nodes:
			# skip the edge designation - we assume it's an arrow
			# find the other node name
			for i in range(start2+3,buf_len):
				if not isspace(<int>buffer[i]):
					break
			
			start2 = i
		
			for i in range(start2+1,buf_len):
				if isspace(<int>buffer[i]) or buffer[i] == ';' or buffer[i] == '[':
					break
			end2 = i
	
			# if any of the positions is buf_len, we're in trouble. At the very least a semi-colon should be at the end of the line.
			if end2 == buf_len:
				raise Exception, 'Edge expected on line %d: %s' % (line_no,buffer)
			
			name1 = node_obj_fxn(buffer[start1:end1])
			name2 = node_obj_fxn(buffer[start2:end2])
			props = None
			
			# find the properties start
			for i in range(end2+1,buf_len):
				if not isspace(<int>buffer[i]):
					break
			
			if i < buf_len and buffer[i] == '[':		
				props = process_properties(buffer,buf_len,i,line_no)
				
			nid1 = None
			nid2 = None
			
			# handle any nodes that didn't have a definition line
			if name1 in G.node_idx_lookup:
				nid1 = G.node_idx_lookup[name1]
			else:
				nid1 = G.add_node(name1)
				
			if name2 in G.node_idx_lookup:
				nid2 = G.node_idx_lookup[name2]
			else:
				nid2 = G.add_node(name2)
				
			G.add_edge_(nid1,nid2,props)
	
	return G
	
cdef process_properties(char* buffer, int buf_len, int i, int line_no):
	"""
	Upon entry, buffer[idx] = '['.  In other words, this function should only
	be called when a property block *does* exist.
	"""
	cdef int sprop,eprop,sstr,estr
	cdef props = None
	while i < buf_len and buffer[i] != ']':
		for i in range(i+1,buf_len):
			if not isspace(<int>buffer[i]):
				break
		sprop = i
		for i in range(sprop+1,buf_len):
			if isspace(<int>buffer[i]) or buffer[i] == '=':
				break
		eprop = i
		# skip the equals sign
		for i in range(eprop,buf_len):
			if buffer[i] == '=':
				break
		for i in range(i+1,buf_len):
			if buffer[i] == '"':
				break
		sstr = i
		for i in range(sstr+1,buf_len):
			if buffer[i] == '"':
				break
		estr = i
		for i in range(estr+1,buf_len):
			if not isspace(<int>buffer[i]):
				break

		# sanity check
		if estr == buf_len:
			raise Exception, 'Syntax error in propertes on line %d: %s' % (line_no,buffer)
									
		prop = buffer[sprop:eprop]
		val = buffer[sstr+1:estr]
		
		if props is None:
			props = {}
			
		props[prop] = val
		
	return props
			
		
		