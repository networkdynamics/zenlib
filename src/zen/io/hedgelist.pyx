#cython: embedsignature=True

from zen.hypergraph cimport HyperGraph

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

cpdef write(HyperGraph G, filename):
	"""
	Write the hypergraph specified in a hyperedgelist format to the filename specified.
	"""
	fh = open(filename,'w')
	
	for nlist in G.edges():
		nlist = [n.replace(' ','_') for n in nlist]
		fh.write(' '.join(nlist) + '\n')
		
	fh.close()
	
cpdef read(char* filename,max_line_len=5000):
	"""
	Read in a network from a file in an edge list format. Hyperedgelist formats are only
	for undirected graphs.
	"""
	G = HyperGraph()

	cdef FILE* fh
	cdef int MAX_LINE_LEN = max_line_len

	# make the string buffer
	str_buffer = '0'*MAX_LINE_LEN

	cdef char* buffer = str_buffer

	# open the file
	fh = fopen(filename,'r')

	if fh == NULL:
		raise Exception, 'Unable to open file %s' % filename

	# read all the lines	
	cdef int start, end
	cdef int line_no = 0
	cdef int buf_len
	cdef int i
	while not feof(fh):
		line_no += 1

		start = 0; end = 0

		fgets(buffer,MAX_LINE_LEN,fh)

		buf_len = len(buffer)

		# check the success
		if not feof(fh) and buffer[buf_len-1] != '\n':
			raise Exception, 'Line %d exceeded maximum line length (%d)' % (line_no,MAX_LINE_LEN)

		# find all the elements
		nlist = []
		i = 0
		while i < buf_len:

			# read a start
			for i in range(i,buf_len):
				if not isspace(<int>buffer[i]):
					break
			start = i

			if start == buf_len:
				break

			for i in range(start+1,buf_len):
				if isspace(<int>buffer[i]):
					break
			end = i
			if end == buf_len-1 and not isspace(<int>buffer[end]):
				end += 1

			n = str(buffer[start:end])
			nlist.append(n)
			i = end + 1

		if len(nlist) > 0:
			G.add_edge(nlist)

	fclose(fh)

	return G