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

__all__ = ['read']

STR_TOK = 'STRING'
INT_TOK = 'INT'
FLOAT_TOK = 'FLOAT'
ID_TOK = 'ID'
SLIST_TOK = '['
ELIST_TOK = ']'

def write(G,filename):
	# Use sphynx style...
	'''Writes graph to file according to graph modelling language
	based on http://www.fim.uni-passau.de/fileadmin/files/lehrstuhl/brandenburg/projekte/gml/gml-technical-report.pdf'''
	
	#if (filename[0:1] != '/'): 		# absolutize filename if necessary
	#filename = os.getcwd() + "/" + filename
	fh = open(filename, 'w')
	
	fh.write('# This is a graph object in gml file format')
	fh.write('\n# produced by the zen graph library')
	fh.write('\ngraph [')
	#fh.write('\n\tcomment \"COMMENT\"')		 #TODO get comment from graph data if exists
	if G.is_directed():
		fh.write('\n\tdirected 1')
	else:
		fh.write('\n\tdirected 0')
	if type(G) == BipartiteGraph:		# to allow recovering proper graph type
		fh.write('\n\tbipartite 1')
	else:
		fh.write('\n\tbipartite 0')		
	#fh.write('\nlabel \"LABEL\"')		# TODO get this from graph obj if exists
	for nidx, nobj, ndata in G.nodes_iter_(obj=True, data=True):
		fh.write('\n\tnode [')
		fh.write('\n\t\tid ' + str(nidx))
		if nobj != None:
			fh.write('\n\t\tlabel ' + write_obj(nobj)) 		# I know you were saying to use name, but gml.read() looks at label for nobj
		if ndata != None:
			fh.write('\n\t\tzen_data ' + write_obj(ndata))
		fh.write('\n\t]')
	
	# iterate over edges
	for eidx, edata, weight in G.edges_iter_(data=True, weight=True):
		fh.write('\n\tedge [')
		fh.write('\n\t\tsource ' + str(G.endpoints_(eidx)[0]))	# for digraphs, assumes endpoints order [source, target]
		fh.write('\n\t\ttarget ' + str(G.endpoints_(eidx)[1]))
		if edata != None:
			fh.write('\n\t\tzen_data ' + write_obj(edata))
		fh.write('\n\t\tweight ' + str(weight))
		fh.write('\n\t]')
	fh.write('\n]')
	fh.close()

def write_obj(obj):
	supported_objs = [str, bool, int, long, float]
	if (type(obj) not in supported_objs) & (obj != None):
		raise ZenException('gml.write() supports node / edge objects: bool, str, Numeric, None')
	if type(obj) in [str, long]:
		obj = '"' + obj.replace('"', '\\"') + '"' # gml specifies integers larger than 32 bit signed must be strings
		# TODO: find a proper escape technique.  GML may not support backslash escape
	# TODO: add arbitrary object serialization here (read() needs unserialize)
	return str(obj)

	
	
def add_token_metadata(token,in_str,lineno):
	
	if in_str:
		return (token,STR_TOK,lineno)
		
	if token == SLIST_TOK:
		return (token,SLIST_TOK,lineno)
	elif token == ELIST_TOK:
		return (token,ELIST_TOK,lineno)
	
	if token.isdigit():
		return (int(token),INT_TOK,lineno)
	else:
		try: # see if it's a float
			return (float(token),FLOAT_TOK,lineno)
		except ValueError:
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
	
	if ttok == STR_TOK or ttok == INT_TOK or ttok== FLOAT_TOK:
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
			
			# enforce types on standard attributes
			if 'name' in node:
				if type(node['name']) != str:
					raise ZenException, 'Node name attribute must be a string (node = %s)' % str(node)
			if 'label' in node:
				if type(node['label']) != str:
					raise ZenException, 'Node label attribute must be a string (node = %s)' % str(node)
			
			# get the node index
			node_idx = None
			if 'id' not in node:
				raise ZenException, 'Node is missing the id attribute (node = %s)' % str(node)
			node_idx = node['id']
			if type(node_idx) != int or node_idx < 0:
				raise ZenException, 'Node id attribute must be a positive integer (node = %s)' % str(node)
				
			# get the node object
			node_obj = node_idx
			if 'name' in node:
				node_obj = node['name']
			elif 'label' in node:
				node_obj = node['label']
				if type(node_obj) != str:
					raise ZenException, 'Node label attribute must be a string (node = %s)' % str(node)
								
			if is_directed:
				G.add_node_x(node_idx,G.edge_list_capacity,G.edge_list_capacity,node_obj,node)
			else:
				G.add_node_x(node_idx,G.edge_list_capacity,node_obj,node)
			
	# add edges
	if 'edge' in graph_data:
		edges = graph_data['edge']
		if type(edges) != list:
			raise ZenException, 'The edge attibute of a graph must be a list'
		
		for edge in edges:
			
			# enforce typed attributes
			edge_idx = None
			if 'id' in edge:
				edge_idx = edge['id']
				if type(edge_idx) != int:
					raise ZenException, 'Edge id attribute must be a positive integer (edge = %s)' % str(edge)
			
			# make sure source and target are specified
			source = None
			target = None
			if 'source' not in edge:
				raise ZenException, 'Edge is missing the source attribute (edge = %s)' % str(edge)
			source = edge['source']
			if type(source) != int or source < 0:
				raise ZenException, 'Edge source attribute must be a positive integer (edge = %s)' % str(edge)
			
			if 'target' not in edge:
				raise ZenException, 'Edge is missing the target attribute (edge = %s)' % str(edge)
			target = edge['target']
			if type(target) != int or source < 0:
				raise ZenException, 'Edge target attribute must be a positive integer (edge = %s)' % str(edge)
				
			weight = 1
			if weight_fxn != None:
				weight = weight_fxn(edge)
			
			if edge_idx != None:
				G.add_edge_x(edge_idx,source,target,edge,weight)
			else:
				G.add_edge_(source,target,edge,weight)
				
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
	
