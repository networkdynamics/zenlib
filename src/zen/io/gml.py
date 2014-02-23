"""
The ``zen.io.gml`` module (available as ``zen.gml``) supports the reading and 
writing network data in the `Graph Modeling Language (GML) 
<http://en.wikipedia.org/wiki/Graph_Modelling_Language>`_.  At present only 
reading GML is supported.

GML is a flexible language for specifying the structure of a network (nodes 
and edges) that can be annotated with arbitrary attributes and information. 
This module provides a full implementation of the GML file format as specified 
in the `technical report 
<http://www.fim.uni-passau.de/fileadmin/files/lehrstuhl/brandenburg/projekte/gml/gml-technical-report.pdf>`_:

.. note::
	Source: Michael Himsolt. *GML: A portable Graph File Format*. Technical 
	Report, Universitat Passau.
	
Functions
---------

.. autofunction:: zen.io.gml.read(fname[,weight_fxn=None])
"""

from zen.exceptions import *
from zen.graph import Graph
from zen.digraph import DiGraph
from zen.bipartite import BipartiteGraph
from gml_codec import BasicGMLCodec, ZenGMLCodec
from gml_interpreter import GMLInterpreter
from gml_tokenizer import GMLTokenizer
import os
import cgi
import re
import codecs
import pdb

__all__ = ['read','write']

VALUE_TOK = t.VALUE_TOK
KEY_TOK = t.KEY_TOK
SLIST_TOK = t.SLIST_TOK
ELIST_TOK = t.ELIST_TOK

DIGITS = tuple(['%d' % x for x in range(10)]) + ('+','-')
DIGITS_AND_QUOTES = DIGITS + ('"',)


# TODO add Creator attribute
# TODO handle > double precision floats
# TODO add Encoder
def write(G,filename, **kwargs):
	"""
	Writes graph to file using Graph Modeling Language (gml).  Node / Edge / 
	Graph objects, if not None, are stored in the `name` attribute, and are 
	restricted to numeric (but not complex), string, and boolean data types, 
	otherwise an exception is raised.  

	Node / Edge / Graph data, if not None, are stored in a zen_data attribute 
	and are similarly restricted.  Support may be added later for serialization
	of arbitrary objects / data and associated to zen graphs.
	
	see 
	http://www.fim.uni-passau.de/fileadmin/files/lehrstuhl/brandenburg/projekte/gml/gml-technical-report.pdf 
	for more info about gml.

	**Args**
		* ``G`` (:py:class:`zen.Graph`, :py:class:'zen.Digraph`, 
			:py:class:`zen.BipartiteGraph): Graph object to be written to 
			file. Hypergraphs are not supported.
	
		* ``filename`` (str): Absolute path for the file to be written.

	**KwArgs**
		* ``write-data`` (bool | (bool, bool)): If 2-tuple of booleans 
			supplied, first indicates whether to write out node data second 
			whether to write out edge data.  If bool provided, it is applied 
			for both node and edge data.

		* ``use-zen-data`` (bool | (bool, bool)): Indicates whether to write 
			out If 2-tuple of booleans supplied, first indicates whether to 
			write out node data second whether to write out edge data.  If bool
			provided, it is applied for both node and edge data. 

	**Returns**:
		* None
	"""

	# Determine the write mode.  There are various options the user can specify
	# which are reconsiled and validated here
	write_node_data, write_edge_data, use_node_zen_data, use_edge_zen_data = resolve_write_mode(kwargs)
	
	# Get the encoder to use.  This call resolves the various ways the user can
	# specify encoding, and does basic checks that the encoder is valid
	enc = resolve_codec(kwargs)

	fh = open(filename, 'w')
	fh.write('# This is a graph object in gml file format\n')
	fh.write('# produced by the zen graph library\n\n')
	fh.write('graph [\n')

	# Describe the encoding method used to generate the file. 
	fh.write('\tZenCodec "%s"\n' % enc.__name__)
	fh.write('\tZenStringEncoder "%s"\n' % enc.encode_str.__name__)

	if G.is_directed():
		fh.write('\tdirected 1\n')

	else:
		fh.write('\tdirected 0\n')

	if type(G) == BipartiteGraph:
		fh.write('\tbipartite 1\n')

	else:
		fh.write('\tbipartite 0\n')		

	# iterate over nodes, writing them to the new gml file
	for nidx, nobj, ndata in G.nodes_iter_(obj=True, data=True):
		fh.write('\tnode [\n')
		fh.write('\t\tid ' + str(nidx) + '\n')

		if nobj != None:
			fh.write(format_zen_data('name', nobj, 2, enc))

		if ndata != None and write_node_data:
			if use_node_zen_data:
				fh.write(format_zen_data('zenData', ndata, 2, enc))

			else:	# expects zenData to be dict
				for key, val in ndata.items():
					fh.write(format_zen_data(key, val, 2, enc))

		fh.write('\t]\n')

	# iterate over edges
	for eidx, edata, weight in G.edges_iter_(data=True, weight=True):
		fh.write('\tedge [\n')

		# for digraphs, assumes endpoints order [source, target]
		fh.write('\t\tsource ' + str(G.endpoints_(eidx)[0]) + '\n')	
		fh.write('\t\ttarget ' + str(G.endpoints_(eidx)[1]) + '\n')
		fh.write('\t\tweight ' + str(weight) + '\n')

		if edata != None and write_edge_data:
			if use_edge_zen_data:
				fh.write(format_zen_data('zenData', edata, 2, enc))

			else:	#expects zenData to be dict
				for key, val in edata.items():
					fh.write(format_zen_data(key, val, 2, enc))

		fh.write('\t]\n')

	fh.write(']\n')
	fh.close()

	
def format_zen_data(keyname, data, tab_depth, encoder, strict=True):
	"""
	Reformats supplied data to use gml.  Enforces restrictions on the types of 
	data that can be written to gml.
	
	**Args**
		* data (bool | int | long | float | str | dict | list): object to be 
			written in gml
		* key (str): key to be used in gml.  Needed here because in gml lists 
			are made by repeating the key in front of each value.
		* tab_depth (int): number of tab characters to add to the beginning 
			of each line for nice formatting.

	**Returns**
		* formatted_data (str): gml representation of data
	"""
	
	# Validation: key names must be strictly alphanumeric
	if re.search('[^a-zA-Z0-9]', keyname) and strict:
		raise ZenException(
			'gml supports only characters from [a-zA-Z0-9] in keys')

	formatted_data = ''
	tabs = '\t' * tab_depth

	if not isinstance(data, (list, dict)):

		encoded_data = encoder.encode(data)

		# Validate encoder output
		if strict:
			try:
				assert(isinstance(encoded_data, basestring))
				assert(encoded_data.startswith(DIGITS_AND_QUOTES))
				encoded_data.encode('ascii')
				if encoded_data.startswith(DIGITS):
					num = float(encoded_data)
					assert(num < 2147483647 or num > -2147483648)

			except AssertionError, UnicodeEncodeError:
				raise ZenException('GML Encoder has violated gml specifications. '\
					'see <http://www.fim.uni-passau.de/fileadmin/files/lehrstuhl/brandenburg/projekte/gml/gml-technical-report.pdf>. \n Use '\
					'gml.write(..., strict=False) to force writing.')

		#TODO: floats should be tested to be within double precision

		# The encoded data is legal for gml. Append extras.
		formatted_data = tabs + keyname + ' ' + encoded_data + '\n'
		
	# Recursive call for lists.  GML represents lists by repeating the
	# key with different values
	elif isinstance(data, list):		
		for val in data:
			formatted_data += format_zen_data(keyname, val, tab_depth, encoder)

	# Recursive call for dicts
	else:
		assert(isinstance(data, dict))
		formatted_data += tabs + keyname + ' [\n'
		for key, val in data.items():
			formatted_data += format_zen_data(key, val, tab_depth + 1, encoder)
		formatted_data += tabs + ']\n'

	return formatted_data


			
		
def add_token_metadata(token,in_str,lineno):

	if in_str:

		if re.match("[-+0-9.]", token[:1]): 	# looks like a number, try it
			try:
				token = int(token)
				return (token, INT_TOK, lineno)
			except ValueError:
				try: 
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

	# TEST: I don't think this will handle leading '+' or '-'...
	if token.isdigit(): 
		return (int(token),INT_TOK,lineno)

	else:
		try: # see if it's a float
			return (float(token),FLOAT_TOK,lineno)
		except ValueError: # Keyname tokens are unquoted strings
			return (token,ID_TOK,lineno)



def parse_key_value_pair(tokens, i, codec):

	key, key_token, key_line_no = tokens[i]

	if key_token != ID_TOK:
		raise ZenException('Line %d: Key must be an identifier, found %s' 
			% (key_line_no, key))

	if (i+1) >= len(tokens):
		raise ZenException('Line %d: Key %s has no value' % (key_line_no, key))

	i,val = parse_value(tokens,i+1, codec)

	return i,key,val


def parse_value(tokens,i, codec):

	val, token_type, lineno = tokens[i]

	if token_type == VALUE_TOK:
		return i+1, codec.decode(val)

	elif token_type == SLIST_TOK:
		return parse_list(tokens,i+1, codec)

	else:
		pdb.set_trace()
		raise ZenException('Line %d: token %s was unexpected' % (lineno,val))


def parse_list(tokens, i, codec):
	start_i = i

	lvals = dict()

	while i < len(tokens) and tokens[i][1] != ELIST_TOK:
		i,key,value = parse_key_value_pair(tokens,i, codec)

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


#def parse_graph_data(tokens, i, codec):
#
#	start_i = i
#	i,key,val = parse_key_value_pair(tokens,i, codec)
#
#	if key != 'graph':
#		raise ZenException('Line %d: Expected a graph block, found %s' 
#			% (tokens[start_i][2],key))
#
#	return i,val


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
				raise ZenException('Node is missing the id attribute (node = %s)' 
					% str(node))

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
						raise ZenException('Node id attribute must be a positive '\
							'integer (node = %s)' % str(node))

				elif key == 'name':
					node_obj = val
					if type(val) != str: # enforce types on standard attributes
						raise ZenException('Node name attribute must be a string '\
							'(node = %s)' % str(node))

				# give preference to 'name' as source of node_obj
				elif key == 'label':
					if node_obj is None: 	
						node_obj = val
					if type(val) != str:
						raise ZenException('Node label attribute must be a string'\
						   ' (node = %s)' % str(node))

				elif key == 'zenData':
					zen_data = val
				
				# node_data is dict of all other attributes
				else:	
					node_data[key] = val 	

			# if zenData is only other attribute aside from those handled above 
			# _set_ to node_data else _append_
			if zen_data is not None:
				if len(node_data) == 0:
					node_data = zen_data
				else:
					node_data['zenData'] = zen_data

			elif len(node_data) == 0:
				node_data = None

			if is_directed:
				G.add_node_x(node_idx, G.edge_list_capacity,
					G.edge_list_capacity, node_obj,node_data)
			else:
				G.add_node_x(node_idx, G.edge_list_capacity, node_obj, node_data)

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
				raise ZenException('Edge is missing the source attribute '\
					'(edge = %s)' % str(edge))

			if 'target' not in edge:
				raise ZenException('Edge is missing the target attribute '\
					'(edge = %s)' % str(edge))

			weight = 1
			edge_idx = None
			zen_data = None
			edge_data = {}

			for key, val in edge.items():
				if key == 'id':
					edge_idx = val
					if type(val) != int:
						raise ZenException('Edge id attribute must be a positive '\
							'integer (edge = %s)' % str(edge))

				elif key == 'source':
					source = val
					if type(val) != int or val < 0:
						raise ZenException('Edge source attribute must be a '\
							'positive integer (edge = %s)' % str(edge))

				elif key == 'target':
					target = val
					if type(val) != int or val < 0:
						raise ZenException('Edge target attribute must be a '\
							'positive integer (edge = %s)' % str(edge))

				elif key == 'weight':
					weight = float(val)

				elif key == 'zenData':
					zen_data = val

				# edge_data is dict of all other attributes
				else: 
					edge_data[key] = val 	

			# give precedence to a weight-getting function if provided
			if weight_fxn != None:
				weight = weight_fxn(edge)

			# if zenData is only other attribute aside from those handled above 
			# _set_ to edge_data else _append_
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
	
	The node's ``id`` attribute is used to specify the node index.  The node's 
	``name`` attribute is preferably used as the node object.  

	However, if the ``name`` attribute is missing and the ``label`` is present,
	then the node's ``label`` attribute will be used as the node object.  If 
	both are missing, then the node id will be used as the node object.
	
	.. note::	
		Currently graph attributes are not supported by the reader.  If 
		encountered, they will simply be skipped over and not added to the
		final graph. This is simply because graph objects don't support 
		arbitrary data yet.
	
	**KwArgs**:
	
		* ``weight_fxn [=None]``: derive weight assignments from edge data.  If
			specified, this function is called with one parameter: the full 
			set of attributes that were specified for the edge.
	"""

	# resolve the codec.  The user can specify the codec in various ways.
	codec = resolve_codec(kwargs)

	# extract keyword arguments
	weight_fxn = kwargs.pop('weight_fxn',None)

	# read the file 
	fh = open(fname,'r')
	gml_str = fh.read()
	fh.close()

	# tokenize the gml string
	tok = GMLTokenizer()
	tokens = tok.tokenize(gml_str)

	# build the graph from list of tokens
	interpreter = GMLInterpreter(codec, tok)
	return intepreter.interpet(tokens)


	i = 0;
	while i < len(tokens):
		# parse top-level key value pairs
		i,key,val = parse_key_value_pair(tokens,i, codec)

		if key == 'graph':
			if type(val) != dict:
				raise ZenException, 'graph attribute must be a list'
			return build_graph(val,weight_fxn)
	
		#TODO: what is this versioning for?
		elif key == 'Version':
			if type(val) != int:
				raise ZenException, 'Version attribute must be an integer'

		#TODO: where is Creator used?
		elif key == 'Creator':
			if type(val) != str:
				raise ZenException, 'Creator attribute must be a string'

	# if we make it here, no graph was loaded
	return None

	
def resolve_write_mode(kwargs):

	# Decide whether to write node and edge data
	# Default is to write both
	write_node_data, write_edge_data = True, True
	if 'write_data' in kwargs:

		# write_data might be a tuple
		if type(kwargs['write_data']) == tuple:
			write_node_data, write_edge_data = kwargs['write_data']

		# or just a single bool
		else:
			write_node_data = write_edge_data = kwargs['write_data']

		#validation:
		if(not isinstance(write_node_data, bool) or not
			isintstance(write_edge_data, bool)):
			raise zenException('write_data keyword argument takes bool or'\
				' 2-tuple of bools. Found: %s (%s)' %(
					write_data, type(write_data)))

	# Decide *how* to write node / edge data to file
	# Default is to use zen_data for both
	use_node_zen_data, use_edge_zen_data = True, True
	if 'use_zen_data' in kwargs:

		if type(kwargs['use_zen_data']) == tuple:
		 	use_node_zen_data, use_edge_zen_data = kwargs['use_zen_data']

		else:
			use_node_zen_data = use_edge_zen_data = kwargs['use_zen_data']

		#validation:
		if(not isinstance(write_node_data, bool) or 
			not isintstance(write_edge_data, bool)):
			raise zenException('write_data keyword argument takes bool or'\
				' 2-tuple of bools. Found: %s (%s)' %(
					write_data, type(write_data)))
	
	return write_node_data, write_edge_data, use_node_zen_data, use_edge_zen_data


def resolve_codec(kwargs):

	# Resolve the data encoder, check if user passed encoder *instance*
	if 'gml_codec' in kwargs:
		enc = kwargs['encoder']
		try:
			assert(isinstance(enc.encode(''), basestring))
			enc.__name__.encode('ascii')
		except (AttributeError, AssertionError) as e:
			raise ZenException('encoder must define encode() to take type '\
				'basestring and return type basestring containing only ascii-'\
				'encodable characers.  It must also provide an ascii-encodable '\
				'__name__ attribute.')

	# Resolve the data encoder, check if user passed encoder by *name*
	elif 'gml_coding' in kwargs:
		if kwargs['encoding'] == 'basic':
			enc = BasicGMLCodec()
		elif kwargs['encoding'] == 'zen':
			enc = ZenGMLCodec()
		else:
			raise ZenException('encoding must be string equal to "basic" or'\
				' "zen"')

	# default encoder
	else:
		enc = BasicGMLCodec()

	# User can also just pass a string-encoder.
	# (The full-fledged gml encoders handle various data types)
	if 'string_encoder' in kwargs:
		str_enc = kwargs['string_encoder']
		try:
			assert(isinstance(str_enc(''), basestring))
		except (TypeError, AssertionError) as e:
			raise ZenException('string_encoder must be a function that takes '\
				'basestring and returns basestring.')

		enc.encode_str = str_enc

	# User can also just pass a string-decoder.
	# (The full-fledged gml encoders handle various data types)
	if 'string_decoder' in kwargs:
		str_dec = kwargs['string_decoder']
		try:
			assert(isinstance(str_dec(''), basestring))
		except (TypeError, AssertionError) as e:
			raise ZenException('string_decoder must be a function that takes '\
				'basestring and returns basestring.')

		enc.decode_str = str_dec

	return enc
