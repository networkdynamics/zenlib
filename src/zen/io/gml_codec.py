from zen.exceptions import ZenException 
from numbers import Number
from collections import Hashable, Iterable
import re
import htmlentitydefs

class Encoder:
	__name__ = 'zen_base_encoder'

	def encode(self, data_to_encode):
		if isinstance(data_to_encode, bool):
			return self.encode_bool(data_to_encode)

		elif isinstance(data_to_encode, Number):
			return self.encode_nbr(data_to_encode)

		elif isinstance(data_to_encode, basestring):
			return self.encode_str(data_to_encode)

		else:
			return self.encode_alt(data_to_encode)


	def basic_encoding(self, data_to_encode):
		return str(data_to_encode)


	def encode_bool(self, data_to_encode):
		return self.basic_encoding(data_to_encode)


	def encode_nbr(self, data_to_encode):
		return self.basic_encoding(data_to_encode)


	def encode_str(self, data_to_encode):
		return self.basic_encoding(data_to_encode)


	def encode_alt(self, data_to_encode):
		raise ZenException('GML codec "%s" provided does not know how to '\
			'encode type %s.  Provide a custom encoder: '\
			'gml.write(..., codec=<your_codec>). See zen.io.gml_codec.py in '\
			'source.' % (self.__name__, type(data_to_encode).__name__))


class Decoder:

	__name__ = 'zen_base_decoder'

	digits = tuple(['%d'%x for x in range(10)]) + ('-','+')

	def decode(self, data_to_decode):
		# Validation
		try:
			assert(isinstance(data_to_decode, basestring))
		except AssertionError:
			raise ZenException('Decoder recieved non-string object: %s(%s) ' 
				% (str(data_to_decode), str(type(data_to_decode))))

		if data_to_decode.startswith('"') or data_to_decode.startswith("'"):
			return self.decode_str(data_to_decode)

		elif data_to_decode.startswith(self.digits):
			return self.decode_nbr(data_to_decode)

		else:
			raise ZenException('Decoder recieved badly formatted value: %s. '\
				'values must start with \'"\' or [-+0-9].' % data_to_decode)
	
	def decode_str(self, data_to_decode):
		return self.remove_quotes(data_to_decode)

	def decode_nbr(self, data_to_decode):
		if '.' in data_to_decode:
			return float(data_to_decode)
		else:
			return int(data_to_decode)

	def remove_quotes(self, data_to_decode):
		return data_to_decode[1:-1]



# stringifies input.  String inputs get wrapped in double quotes
# Numbers are passed through non-enquoted
class BasicGMLCodec(Encoder, Decoder):

	__name__ = 'basic'

	def basic_encoding(self, data_to_encode):
		return enquote(Encoder.basic_encoding(self, data_to_encode))

	def encode_bool(self, data_to_encode):
		return str(1 if data_to_encode else 0)

	def encode_str(self, data_to_encode):
		# escape ampersands
		data_to_encode = data_to_encode.replace('&', '&amp;')

		# escape non-ascii characters, then return
		return enquote(data_to_encode.encode('ascii', 'xmlcharrefreplace'))

	def encode_nbr(self, data_to_encode):
		if data_to_encode < 2147483647 and data_to_encode > -2147483648:	
			return str(data_to_encode)

		else:
			return enquote(str(data_to_encode))

	# remove quotes, check if its a string-encoded int, and unescape xml 
	# character references
	def decode_str(self, data_to_decode):
		unquoted = self.remove_quotes(data_to_decode)

		# integers > 32 bits get encoded as strings.  Try reading as int.
		if unquoted.startswith(self.digits):
			try:
				return self.decode_nbr(unquoted)
			except ValueError:
				pass 

		# otherwise its a normal string.  Unescape any xml entities
		return unescape(unquoted)




class ZenGMLCodec(BasicGMLCodec):

	__name__ = 'zen'

	def encode_str(self, data_to_encode):
		# escape leading slash, which is used to prevent collision between
		# digit-containing strings and encoding of large numbers using strings
		if data_to_encode.startswith('\\'):
			data_to_encode = '\\' + data_to_encode

		elif data_to_encode.startswith('#'):
			data_to_encode = '\\' + data_to_encode

		elif data_to_encode.startswith(self.digits):
			data_to_encode = '#' + data_to_encode

		return BasicGMLCodec.encode_str(self, data_to_encode)


	def decode_str(self, data_to_decode):
		unquoted = self.remove_quotes(data_to_decode)

		# Check for escapage in prefix
		if unquoted.startswith(('\\', '#')):
			unprefixed = unquoted[1:]

		# If not escapage in prefix, try parsing like a number
		else:
			unprefixed = unquoted
			if unprefixed.startswith(self.digits):

				# try decoding as a number.  If not, decode as a string
				try:
					return self.decode_nbr(unprefixed)
				except ValueError:
					pass
		
		return unescape(unprefixed)


def enquote(string):
	return '"' + string + '"'

##
# Removes HTML or XML character references and entities from a text string.
# Credit: Fredrik Lundh <http://effbot.org/zone/re-sub.htm#unescape-html>
#
# @param text The HTML (or XML) source text.
# @return The plain text, as a Unicode string, if necessary.
# 
def unescape(text):
	def fixup(m):
		text = m.group(0)
		if text[:2] == "&#":
			# character reference
			try:
				if text[:3] == "&#x":
					return unichr(int(text[3:-1], 16))
				else:
					return unichr(int(text[2:-1]))
			except ValueError:
				pass
		else:
			# named entity
			try:
				text = unichr(htmlentitydefs.name2codepoint[text[1:-1]])
			except KeyError:
				pass
		return text # leave as is
	return re.sub("&#?\w+;", fixup, text)
