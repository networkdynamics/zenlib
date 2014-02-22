from zen.exceptions import *
from gml_codec import BasicGMLCodec, ZenGMLCodec
import re
import codecs
import pdb

KEY_TOK = 0
VAL_TOK = 1
SLIST_TOK = 2
ELIST_TOK = 3

class Tokenizer(object):


	def __init__(self):

		self.refresh()
		self.restart()

		# regex testers
		self.letters = re.compile(r'[a-zA-Z]')
		self.number_before_decimal = re.compile(r'[.0-9]')
		self.number_after_decimal = re.compile(r'[0-9]')
		self.number_starter = re.compile(r'[-+.0-9]')
		self.non_alpha = re.compile(r'[^a-zA-Z]')


	def refresh(self):

		# tokenizer state
		self.current_token = ''
		self.expected_token = 'key'
		self.in_token = False
		self.in_quotes = False
		self.after_decimal = False
		self.do_break = False

	def restart(self):
		# token accumulater
		self.tokens = []


	def _end_current_token_then_expect(self, next_expected_token_type):
		'''
		Ends the currently aggregating token, and puts it onto the list of
		tokens.  Refreshes the tokenizers state to expect the next token
		next_expected_token_type should be string either 'key' or 'value'
		'''

		self.refresh()
		self.expected_token = next_expected_token_type


	def tokenize(self, gml_string):
		'''
		this is the main public method to be called.  It accepts a gml string
		and returns a list of tokens.  Tokens are tuples containing substrings,
		a token type (string) and the line number from which the token was 
		taken.

		In this method we see how the state of the tokenizer maps to behavior.
		processing is char by char, and the processing of a char is delegated
		to a private method based on the state of the tokenizer.
		'''

		self.refresh()
		self.restart()

		# Iterate over all the lines in the string
		for line_num, line in enumerate(
			split_but_keep_delimiter(gml_string, '\n')):

			# Iterate over all the characters in the line
			for col_num, char in enumerate(line):

				# A tree of binary conditionals deligates the work to the
				# right handler.  Note that aggregation of the tokens, and
				# state changes happen as side effects in the handlers
				if not self.in_token:

					if self.expected_token == 'key':

						if self.in_quotes:
							raise ZenException(
								'Internal tokenizer state conflict: cannot '\
								'be in quotes but not in token.')

						else:
							self._handle_notintoken_expectingkey(
								char, line_num, col_num)
							if self.do_break: 
								self.do_break = False
								break


					else: # self.expected_token == 'val'

						if self.in_quotes:
							raise ZenException(
								'Internal tokenizer state conflict: cannot '\
								'be in quotes but not in token.')

						else:
							self._handle_notintoken_expectingval(
								char, line_num, col_num)
							if self.do_break: 
								self.do_break = False
								break

				else: # self.in_token is True

					if self.expected_token == 'key':

						if self.in_quotes:
							raise ZenException(
								'Internal tokenizer state conflict: a '\
								'key cannot be a string literal.')

						else:
							self._handle_inkey_notquoted(
								char, line_num, col_num)
							if self.do_break: 
								self.do_break = False
								break

					else: # self.expected_token == 'val'

						if self.in_quotes:
							self._handle_inval_quoted(
								char, line_num, col_num)
							if self.do_break: 
								self.do_break = False
								break

						else: # not self.in_quotes

							if not self.after_decimal:
								self._handle_inval_notquoted_beforedec(
									char, line_num, col_num)
								if self.do_break: 
									self.do_break = False
									break

							else: # self.after_decimal is True
								self._handle_inval_notquoted_afterdec(
									char, line_num, col_num)
								if self.do_break: 
									self.do_break = False
									break

		if self.in_token:
			raise ZenException('Unexpected end of file while reading token '\
				'%s.' 
				% (repr(self.current_token)))

		return self.tokens


	def _handle_notintoken_expectingkey(self, char, line_num, col_num):

		if char.isspace():
			return

		elif char == '#':
			self.do_break = True
			return

		elif char == '"':
			raise ZenException('Bad GML format: expecting key but found '\
				'string literal (line %d, col %d)' % (line_num, col_num))

		elif char == '[':
			raise ZenException('Bad GML format: expecting key but found '\
				'"[" (line %d, col %d)' % (line_num, col_num))

		elif char == ']':

			# start a new ELIST_TOK
			self.in_token = True
			self.current_token += char

			# immediately store ond close the ELIST_TOK
			self.tokens.append((self.current_token, ELIST_TOK, line_num))
			self._end_current_token_then_expect('key')
			return


		elif self.letters.search(char):

			# start new KEY_TOK
			self.in_token = True
			self.current_token += char
			return

		else:
			raise ZenException('Bad GML format: keys must contain only '\
				'a-z or A-Z.  Found %s (line %d, col %d)' 
				% (repr(char), line_num, col_num))


	def _handle_notintoken_expectingval(self, char, line_num, col_num):
		if char.isspace():
			return

		elif char == '#':
			self.do_break = True
			return

		elif char == '"':

			# starts a new VAL_TOK in quotes
			self.in_token = True
			self.in_quotes = True
			self.current_token += char
			return

		elif char == '[':

			# starts a new SLIST_TOK
			self.in_token = True
			self.current_token += char

			# immediately stores and closes SLIST_TOK
			self.tokens.append((self.current_token, SLIST_TOK, line_num))
			self._end_current_token_then_expect('key')
			return

		elif char == ']':
			raise ZenException('Bad GML format: expecting value token but '\
				'found "]" (line %d, col %d)' % (line_num, col_num))

		elif self.letters.search(char):
			raise ZenException('Bad GML format: expecting value token but '\
				'found alpha char %s (line %d, col %d).  Values must be '\
				'numeric or (enquoted) string literals ' 
				% (repr(char), line_num, col_num))

		elif self.number_starter.search(char):

			# starts a new VAL_TOK
			self.in_token = True
			self.current_token += char
			return

		else:
			raise ZenException('Bad GML format: expecting value token but '\
				'found %s. Values must be numeric or (enquoted) string '\
				'literals (line %d, col %d)' % (repr(char), line_num, col_num))


	def _handle_inkey_notquoted(self, char, line_num, col_num):

		if char.isspace():

			# store and close the current token
			self.tokens.append((self.current_token, KEY_TOK, line_num))

			self._end_current_token_then_expect('val')
			return

		elif char == '#':

			# store and close current token, and break out of current line
			self.tokens.append((self.current_token, KEY_TOK, line_num))
			self._end_current_token_then_expect('val')
			self.do_break = True
			return

		elif self.letters.search(char):

			# continue the current token
			self.current_token += char
			return

		else:
			raise ZenException('Bad GML format: found non-alpha char %s '\
				'while processing key token (line %d, col %d)' 
				% (repr(char), line_num, col_num))
		

	def _handle_inval_quoted(self, char, line_num, col_num):

		if char == '"':

			# store and close the current token
			self.current_token += char
			self.tokens.append((self.current_token, VAL_TOK, line_num))
			self._end_current_token_then_expect('key')
			return

		else:

			# continue the current token
			self.current_token += char
			return


	def _handle_inval_notquoted_beforedec(self, char, line_num, col_num):

		if char.isspace():

			# store and close the current token
			self.tokens.append((self.current_token, VAL_TOK, line_num))
			self._end_current_token_then_expect('key')
			return

		if char == '#':

			# store and close the current token, break to the next line
			self.tokens.append((self.current_token, VAL_TOK, line_num))
			self._end_current_token_then_expect('key')
			self.do_break = True
			return

		if self.number_before_decimal.search(char):

			# continue the current token
			self.current_token += char
			return

		else:
			raise ZenException('Bad GML format: found invalid char %s '\
				'while processing numeric value token (line %d, col %d)'
				% (repr(char), line_num, col_num))


	def _handle_inval_notquoted_afterdec(self, char, line_num, col_num):
		
		if char.isspace():

			# store and close the current token
			self.tokens.append((self.current_token, VAL_TOK, line_num))
			self._end_current_token_then_expect('key')
			return

		if char == '#':

			# store and close the current token, break to the next line
			self.tokens.append((self.current_token, VAL_TOK, line_num))
			self._end_current_token_then_expect('key')
			self.do_break = True
			return

		if self.number_after_decimal.search(char):

			# continue the current token
			self.current_token += char
			return

		else:
			raise ZenException('Bad GML format: found invalid char %s '\
				'while processing numeric value token (line %d, col %d)'
				% (repr(char), line_num, col_num))
	


					


def split_but_keep_delimiter(string, delimiter):
	return [substring + delimiter for substring in string.split(delimiter)]

