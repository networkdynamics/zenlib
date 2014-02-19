"""
The ``zen.io.gml`` module (available as ``zen.gml``) supports the reading and writing network data in the `Graph Modeling Language (GML) <http://en.wikipedia.org/wiki/Graph_Modelling_Language>`_.  At present only reading GML is supported.

GML is a flexible language for specifying the structure of a network (nodes and edges) that can be annotated with arbitrary attributes and information. This module provides a full implementation of the GML file format as specified in the `technical report <http://www.fim.uni-passau.de/fileadmin/files/lehrstuhl/brandenburg/projekte/gml/gml-technical-report.pdf>`_:

.. note::
	Source: Michael Himsolt. *GML: A portable Graph File Format*. Technical Report, Universitat Passau.
	
Functions
---------

.. autofunction:: zen.io.gml.read(fname[,weight_fxn=None])
"""

from zen.exceptions import *
from zen.graph import Graph
from zen.digraph import DiGraph
from zen.bipartite import BipartiteGraph
import os
import cgi
import re

__all__ = ['read','write']

STR_TOK = 'STRING'
INT_TOK = 'INT'
FLOAT_TOK = 'FLOAT'
BOOL_TOK = "BOOL"
ID_TOK = 'ID'
SLIST_TOK = '['
ELIST_TOK = ']'

def write(G,filename, **kwargs):
	"""
	Writes graph to file using Graph Modeling Language (gml).  Node / Edge / Graph objects, if not None, are stored in the `name` 
	attribute, and are restricted to numeric (but not complex), string, and boolean data types, otherwise an exception is raised.  
	Node / Edge / Graph data, if not None, are stored in a zen_data attribute and are similarly restricted.
	Support may be added later for serialization of arbitrary objects / data and associated to zen graphs.
	
	see http://www.fim.uni-passau.de/fileadmin/files/lehrstuhl/brandenburg/projekte/gml/gml-technical-report.pdf for more
	info about gml.

	**Args**
		* ``G`` (:py:class:`zen.Graph`, :py:class:'zen.Digraph`, :py:class:`zen.BipartiteGraph): Graph object to be
			written to file. Hypergraphs are not supported.
	
		* ``filename`` (str): Absolute path for the file to be written.
	**KwArgs**
		* ``write-data`` (bool | (bool, bool)): If 2-tuple of booleans supplied, first indicates whether to write out node data
				second whether to write out edge data.  If bool provided, it is applied for both node and edge data.
		* ``use-zen-data`` (bool | (bool, bool)): Indicates whether to write out If 2-tuple of booleans supplied, first indicates whether to write out node data
						second whether to write out edge data.  If bool provided, it is applied for both node and edge data. 
	**Returns**:
		* None
	"""

	# Decide whether to write node and edge data
	if 'write_data' in kwargs:
		if type(kwargs['write_data']) == tuple:
			write_node_data, write_edge_data = kwargs['write_data']
		elif type(kwargs['write_data']) == bool:
			write_node_data = write_edge_data = kwargs['write_data']
		# else raise zenException?  Not sure how far to take verification...
			
	else:
		write_node_data, write_edge_data = True, True

	# Decide *how* to write node / edge data to file
	if 'use_zen_data' in kwargs:
		if type(kwargs['use_zen_data']) == tuple:
		 	use_node_zen_data, use_edge_zen_data = kwargs['use_zen_data']
		else:
			use_node_zen_data = use_edge_zen_data = kwargs['use_zen_data']
		#else raise zenException?s
		
	else: 
		use_node_zen_data = use_edge_zen_data = True
		
	fh = open(filename, 'w')
	fh.write('# This is a graph object in gml file format\n')
	fh.write('# produced by the zen graph library\n\n')
	fh.write('graph [\n')

	if G.is_directed():
		fh.write('\tdirected 1\n')
	else:
		fh.write('\tdirected 0\n')
	
	if type(G) == BipartiteGraph:
		fh.write('\tbipartite 1\n')
	else:
		fh.write('\tbipartite 0\n')		
	
	for nidx, nobj, ndata in G.nodes_iter_(obj=True, data=True):
		fh.write('\tnode [\n')
		fh.write('\t\tid ' + str(nidx) + '\n')
		
		if nobj != None:
			fh.write(format_zen_data(nobj, 'name', 2))
		
		if ndata != None and write_node_data:
			if use_node_zen_data:
				fh.write(format_zen_data(ndata, 'zenData', 2))
			
			else:	# expects zenData to be dict
				for key, val in ndata.items():
					fh.write(format_zen_data(val, key, 2))
			
		fh.write('\t]\n')
	
	# iterate over edges
	for eidx, edata, weight in G.edges_iter_(data=True, weight=True):
		fh.write('\tedge [\n')
		fh.write('\t\tsource ' + str(G.endpoints_(eidx)[0]) + '\n')	# for digraphs, assumes endpoints order [source, target]
		fh.write('\t\ttarget ' + str(G.endpoints_(eidx)[1]) + '\n')
		fh.write('\t\tweight ' + str(weight) + '\n')
		
		if edata != None and write_edge_data:
			if use_edge_zen_data:
				fh.write(format_zen_data(edata, 'zenData', 2))
			
			else:	#expects zenData to be dict
				for key, val in edata.items():
					fh.write(format_zen_data(val, key, 2))
		
		fh.write('\t]\n')
	
	fh.write(']\n')
	fh.close()

	
def format_zen_data(data, keyname, tab_depth=0):
	"""
	Reformats supplied data to use gml.  Enforces restrictions on the types of data that can be written to gml.
	
	**Args**
		* data (bool | int | long | float | str | dict | list): object to be written in gml
		* key (str): key to be used in gml.  Needed here because in gml lists are made by repeating the key in front of each value.
		* tab_depth (int): number of tab characters to add to the beginning of each line for nice formatting.

	**Returns**
		* formatted_data (str): gml representation of data
	"""
	
	formatted_data = ''
	tabs = '\t' * tab_depth
	if re.search('[^a-zA-Z0-9]', keyname):
		raise ZenException('gml supports only characters from [a-zA-Z0-9] in keys')
	
	if type(data) == bool: # booleans are recorded as strings.  They have to be detected on read
		formatted_data += tabs + keyname + ' "' + str(data) + '"\n'
	
	elif type(data) == int or type(data) == long:		
		if data > 2147483647 or data < -2147483648:	# gml specifies integers larger than 32 bit signed must be strings
			formatted_data += tabs + keyname + ' "' + str(data) + '"\n'
		else:
			formatted_data += tabs + keyname + ' ' + str(data) + '\n'

	elif type(data) == float:
		formatted_data += tabs + keyname + ' ' + str(data) + '\n'
		
	elif type(data) == str:
		data = cgi.escape(data, quote=True)		# escapes xml special characters and quotes
		data = data.decode('utf-8').encode('iso_8859_1', 'xmlcharrefreplace')		# escapes non-ISO-8859-1 chars
		formatted_data += tabs + keyname + ' "' + data.decode('utf-8').encode('iso_8859_1', 'xmlcharrefreplace') + '"\n'

	elif type(data) == list:		# this may seem odd, because gml uses repeated keys to indicate a list
		for val in data:
			formatted_data += format_zen_data(val, keyname, tab_depth)
	
	elif type(data) == dict:
		formatted_data += tabs + keyname + ' [\n'
		for key, val in data.items():
			formatted_data += format_zen_data(val, key, tab_depth + 1)
		formatted_data += tabs + ']\n'
	
	else:
		raise ZenException('gml.write() supports objects: bool, str, Numeric (not complex), None, and dicts or lists containing only such types (nesting allowed)')
	
	return formatted_data
	
def decode_xml_entities(s):
	# D: I read through encoding info and couldn't find the functions to undo the substitution of xml entities
	# so I just wrote it.  It took me a lot less time than I spent searching... go figure
	s_decode = u''
	
	i = 0
	while  i < len(s):

		if s[i] == '&':
			i = i+1
			code_point = ''
			
			while s[i] != ';':
				code_point += s[i]
				i = i+1
			
			if code_point == 'quot':
				s_decode += '"'
			elif code_point == 'lt':
				s_decode += '<'
			elif code_point == 'gt':
				s_decode += '>'
			elif code_point == 'amp':
				s_decode += '&'
			elif code_point[:1] == '#':
				s_decode += unichr(int(code_point[1:]))
			else:
				raise ZenException, 'Cannot recognize the xml entity &%s;' % code_point
			
			i = i+1

		else:
			s_decode += s[i]
			i = i+1

	return s_decode.encode('utf-8')
			
		
def add_token_metadata(token,in_str,lineno):
	
	if in_str:
		if re.match("[-+0-9.]", token[:1]): 	# looks like a number, try it
			try:
				token = int(token)
				return (token, INT_TOK, lineno)
			except ValueError:
				token = float(token)
				return (token, FLOAT_TOK, lineno)
			except ValueError:
				return (token, STR_TOK, lineno)
		
		elif token == "True":		# treat as boolean
			token = True
			return (token, BOOL_TOK, lineno)

		elif token == "False":
			token = False
			return (token, BOOL_TOK, lineno)
			
		else:	
			return (token,STR_TOK,lineno)
				
	if token == SLIST_TOK:
		return (token,SLIST_TOK,lineno)
	elif token == ELIST_TOK:
		return (token,ELIST_TOK,lineno)
	
	if token.isdigit(): # TEST: I don't think this will handle leading '+' or '-'...
		return (int(token),INT_TOK,lineno)
	else:
		try: # see if it's a float
			return (float(token),FLOAT_TOK,lineno)
		except ValueError: # Keyname tokens are unquoted strings
			return (token,ID_TOK,lineno)

def tokenize(fh):
	tokens = []
	
	for lineno,line in enumerate(fh):
		line = line.strip()
		
		token = None
		in_str = False
		for c in line:
			# comments kill the rest of the line
			if not in_str and c == '#':
				break
				
			if not in_str and c.isspace(): # unquoted spaces end a token
				tokens.append(add_token_metadata(token,in_str,lineno))
				token = None
			elif c == '"': # quotes start and end strings
				if in_str: # string is ending
					tokens.append(add_token_metadata(token,in_str,lineno))
					in_str = False
					token = None
				else: # a string is starting
					if token != None:
						tokens.append(add_token_metadata(token,in_str,lineno))
					token = ''
					in_str = True
			elif in_str:
				token += c
			elif c == '[' or c == ']': # unquoted braces are tokens
				if token != None:
					tokens.append(add_token_metadata(token,in_str,lineno))
				tokens.append(add_token_metadata(c,in_str,lineno))
				token = None
			else:
				if token is None:
					token = c
				else:
					token += c
					
		# strings can't be multlined
		if in_str:
			raise ZenException, "Line %d: A string can't span lines in a GML file" % lineno
		
		# handle unfinished tokens	
		if token != None:
			tokens.append(add_token_metadata(token,in_str,lineno))
			
	return tokens
	
def parse_key_value_pair(tokens,i):
	
	key,ktok,kline = tokens[i]
	
	if ktok != ID_TOK:
		raise ZenException, 'Line %d: Key must be an identifier, found %s' % (kline,key)
	
	if (i+1) >= len(tokens):
		raise ZenException, 'Line %d: Key %s has no value' % (kline,key)
	
	i,val = parse_value(tokens,i+1)
	
	return i,key,val
	
def parse_value(tokens,i):
	
	val,ttok,lineno = tokens[i]
	
	if ttok == INT_TOK or ttok == FLOAT_TOK or ttok == BOOL_TOK:
		return i+1,val
	elif ttok == STR_TOK: # restore characters escaped by xmlcharrefreplace (need to use a unicode, UTF-8 string)
		val = decode_xml_entities(val)
		return i+1,val
	elif ttok == SLIST_TOK:
		return parse_list(tokens,i+1)
	else:
		raise ZenException,'Line %d: token %s was unexpected' % (lineno,val)
		
def parse_list(tokens,i):
	start_i = i
	
	lvals = dict()
	
	while i < len(tokens) and tokens[i][1] != ELIST_TOK:
		i,key,value = parse_key_value_pair(tokens,i)
		
		if key in lvals: # if the same key is used over and over again, it's indicating a list of values
			curr_val = lvals[key]
			if type(curr_val) == list:
				curr_val.append(value)
			else:
				lvals[key] = [curr_val,value]
		else:
			lvals[key] = value
		
	if i == len(tokens):
		raise ZenException, 'Line %d: List is not closed' % tokens[start_i][2]
	
	return i+1,lvals
	
def parse_graph_data(tokens,i):
	
	start_i = i
	i,key,val = parse_key_value_pair(tokens,i)
	
	if key != 'graph':
		raise ZenException, 'Line %d: Expected a graph block, found %s' % (tokens[start_i][2],key) 
	
	return i,val

def build_graph(graph_data,weight_fxn):
	
	is_directed = False
	if 'directed' in graph_data:
		is_directed = graph_data['directed'] == 1
	
	# TODO detect if graphs are bipartite and support that
		
	G = None
	if is_directed:
		G = DiGraph()
	else:
		G = Graph()
		
	# TODO: Load graph attributes
		
	# add nodes
	if 'node' in graph_data:
		nodes = graph_data['node']
		if type(nodes) != list:
			raise ZenException, 'The node attribute of a graph must be a list'
			
		for node in nodes:	
			# node must have an 'id'
			if 'id' not in node:
				raise ZenException, 'Node is missing the id attribute (node = %s)' % str(node)
			node_idx = node['id']
				
			# collect and verify all the node properties
			standard_keys = set(['id', 'name', 'zenData'])
			node_data = {}
			node_obj = None
			zen_data = None
			for key, val in node.items():
				if key == 'id':
					node_idx = val
					if type(val) != int or val < 0:
						raise ZenException, 'Node id attribute must be a positive integer (node = %s)' % str(node)

				elif key == 'name':
					node_obj = val
					if type(val) != str: # enforce types on standard attributes
						raise ZenException, 'Node name attribute must be a string (node = %s)' % str(node)

				elif key == 'label':
					if node_obj is None: 	# give preference to 'name' as source of node_obj
						node_obj = val
					if type(val) != str:
						raise ZenException, 'Node label attribute must be a string (node = %s)' % str(node)

				elif key == 'zenData':
					zen_data = val
				
				else:	
					node_data[key] = val 	# node_data is dict of all other attributes

			# if zenData is only other attribute aside from those handled above _set_ to node_data else _append_
			if zen_data is not None:
				if len(node_data) == 0:
					node_data = zen_data
				else:
					node_data['zenData'] = zen_data
			
			elif len(node_data) == 0:
				node_data = None
								
			if is_directed:
				G.add_node_x(node_idx,G.edge_list_capacity,G.edge_list_capacity,node_obj,node_data)
			else:
				G.add_node_x(node_idx,G.edge_list_capacity,node_obj,node_data)
			
	# add edges
	if 'edge' in graph_data:
		edges = graph_data['edge']
		if type(edges) != list:
			raise ZenException, 'The edge attibute of a graph must be a list'
		
		for edge in edges:
			
			# make sure source and target are specified
			source = None
			target = None
			if 'source' not in edge:
				raise ZenException, 'Edge is missing the source attribute (edge = %s)' % str(edge)
			
			if 'target' not in edge:
				raise ZenException, 'Edge is missing the target attribute (edge = %s)' % str(edge)
			
			weight = 1
			edge_idx = None
			zen_data = None
			edge_data = {}
			
			for key, val in edge.items():
				if key == 'id':
					edge_idx = val
					if type(val) != int:
						raise ZenException, 'Edge id attribute must be a positive integer (edge = %s)' % str(edge)
				
				elif key == 'source':
					source = val
					if type(val) != int or val < 0:
						raise ZenException, 'Edge source attribute must be a positive integer (edge = %s)' % str(edge)
				
				elif key == 'target':
					target = val
					if type(val) != int or val < 0:
						raise ZenException, 'Edge target attribute must be a positive integer (edge = %s)' % str(edge)
				
				elif key == 'weight':
					weight = float(val)
				
				elif key == 'zenData':
					zen_data = val
			
				else: 
					edge_data[key] = val 	# edge_data is dict of all other attributes
			
			# give precedence to a weight-getting function if provided
			if weight_fxn != None:
				weight = weight_fxn(edge)
			
			# if zenData is only other attribute aside from those handled above _set_ to edge_data else _append_
			if zen_data is not None:
				if len(edge_data) == 0:
					edge_data = zen_data
				else:
					edge_data['zenData'] = zen_data

			elif len(edge_data) == 0:
				edge_data = None;
		
			if edge_idx != None:
				G.add_edge_x(edge_idx,source,target,edge_data,weight)
			else:
				G.add_edge_(source,target,edge_data,weight)
			
			
				
	return G
				
def read(fname,**kwargs):
	"""
	Read GML-formatted network data stored in file named ``fname``.
	
	The node's ``id`` attribute is used to specify the node index.  The node's ``name`` attribute is preferably used as the node object.  
	However, if the ``name`` attribute is missing and the ``label`` is present, then the node's ``label`` attribute will be used as the 
	node object.  If both are missing, then the node id will be used as the node object.
	
	.. note::	
		Currently graph attributes are not supported by the reader.  If encountered, they will simply be skipped over and not added to the
		final graph. This is simply because graph objects don't support arbitrary data yet.
	
	**KwArgs**:
	
		* ``weight_fxn [=None]``: derive weight assignments from edge data.  If specified, this function is called with one parameter: the
		  full set of attributes that were specified for the edge.
	"""
	
	# extract keyword arguments
	weight_fxn = kwargs.pop('weight_fxn',None)
	
	# tokenize the data
	fh = open(fname,'r')
	tokens = tokenize(fh)
	#print tokens
	fh.close()
	
	i = 0
	while i < len(tokens):
		# parse top-level key value pairs
		i,key,val = parse_key_value_pair(tokens,i)
		
		if key == 'graph':
			if type(val) != dict:
				raise ZenException, 'graph attribute must be a list'
			return build_graph(val,weight_fxn)
		elif key == 'Version':
			if type(val) != int:
				raise ZenException, 'Version attribute must be an integer'
		elif key == 'Creator':
			if type(val) != str:
				raise ZenException, 'Creator attribute must be a string'
				
	# if we make it here, no graph was loaded
	return None
	
