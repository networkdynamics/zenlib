'''
	GMLInterpreter maps a list of gml tokens into an in-memory representation 
	using python dicts and lists.  This enforces the grammar of GML and 
	thows a ZenException on grammar violations.  It needs to receive a token
	list from a tokenizer, and expects that the tokens themselves have been
	validated (e.g. strings are properly closed, gml key names are strictly
	alphabetic.

	Such token-validation, however, could been disabled in the tokenizer.

	The GMLInterpreter constructor takes a tokenizer (the one which produced 
	the list of tokens), and a GMLCodec.  The tokenizer is taken in order to 
	verify that it defines token-types that the GMLInterpreter expects
'''

from zen.exceptions import *
from zen.graph import Graph
from zen.digraph import DiGraph
from zen.bipartite import BipartiteGraph
import pdb


class GMLInterpreter(object):

	def __init__(self, gml_codec, tokenizer):
		self.codec = gml_codec

		# ensure that the tokenizer defines the token-types that the 
		# GMLInterpreter expects:
		try:
			self.VAL_TOK = tokenizer.VAL_TOK
			self.KEY_TOK = tokenizer.KEY_TOK
			self.SLIST_TOK = tokenizer.SLIST_TOK
			self.ELIST_TOK = tokenizer.ELIST_TOK

		except AttributeError as e:
			raise ZenException('Could not build GMLInterpreter: the '\
				'tokenizer received does not define the expected toke types')


	def restart(self):
		self.ptr = 0				# pointer to tokens


	def evaluate_list(self, tokens, depth):

		interpretation = {}
		while self.ptr  < len(tokens):

			# Validate the next token's type
			token_type = tokens[self.ptr][1]

			# Usually, the next token should be a key token 
			if token_type == self.KEY_TOK:
				pass

			# if not, then it must be an end-list token
			elif token_type == self.ELIST_TOK:
				
				# an end-list token is only valid when depth > 0
				if depth > 0:
					pass

				else:
					raise ZenException('Unexpected end of list token "]" '\
						'at line %d' % tokens[self.ptr][2])

			# We shouldn't get an start-list or value token here
			elif token_type == self.SLIST_TOK or token_type == self.VAL_TOK:
				raise ZenException('Unexpected token %s at line %d.' 
					% (repr(tokens[self.ptr][0]), tokens[self.ptr][2]))


			# If the token isn't of any of the above types, it's unrecognized
			else:
				raise ZenException('Received unrecognized token type: '\
					'%s.  The tokenizer is not compatible with this '\
					'interpreter.' % repr(tokens[self.ptr]))

			# We have a valid token
			next_key = tokens[self.ptr]

			# if it is a list-end token, advance ptr and return the 
			# interpretation 
			if token_type == self.ELIST_TOK:
				self.ptr += 1
				return interpretation

			# Otherwise this is a key-token add a new key-value pair
			else:
				new_key = tokens[self.ptr][0]

				# verify that there is another token for the value
				if self.ptr + 1 >= len(tokens):
					raise ZenException('Unexpected end of file after '\
						'dangling token %s' % repr(tokens[-1][0]))

				# if the next token is a start-list token, recursively 
				# interpret that list. (GML lists are like python dicts)
				if tokens[self.ptr + 1][1] == self.SLIST_TOK:
					self.ptr += 2
					new_val = self.evaluate_list(tokens, depth + 1)

				# Otherwise we have a primitive value.  
				# Interpret it with the help of the codec
				else:

					# Make sure that we did get a VAL_TOK here
					if tokens[self.ptr + 1][1] != self.VAL_TOK:
						raise ZenException('Expected a value-token but found '\
							'%s on line %d' % (tokens[self.ptr + 1][0], 
							tokens[self.ptr + 1][2]))

					new_val = self.codec.decode(tokens[self.ptr + 1][0])
					self.ptr += 2

				# Check if this key has occured before, if so, this represents
				# GML's encoding of what would be called a list in python
				if new_key in interpretation:

					# have we already made this into a list? If so just append
					if isinstance(interpretation[new_key], list):
						interpretation[new_key].append(new_val)

					# if not, make it a list, and put the new value
					else:
						old_val = interpretation[new_key]
						interpretation[new_key] = [old_val, new_val]

				# Otherwise this is a simple key-value pair
				else:
					interpretation[new_key] = new_val

		# if we get here, we've finished consuming the tokens
		# we must not be in a recursive call
		if depth > 0:
			raise ZenException('Unexpected end of file while parsing a list '\
				'perhaps a "]" is missing.')

		# Also, there shouldn't be a single token left (tokens are parsed in
		# pairs)
		if self.ptr < len(tokens):
			raise ZenException('Unexpected end of file after dangling token '\
				'%s' % repr(tokens[-1][0]))

		# All is good, we have a valid interpretation of the document
		return interpretation
		

	def interpret(self, tokens):

		self.restart()
		return self.evaluate_list(tokens, 0)



