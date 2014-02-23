from zen import *
import unittest
import os
import os.path as path
import tempfile

class GMLTokenizerCase(unittest.TestCase):
		
	tok = gml_tokenizer.GMLTokenizer()
	codec = gml_codec.BasicGMLCodec()
	interp = gml_interpreter.GMLInterpreter(codec, tok)

	def test_basic_correct(self):

		tokens = [
			('keyOne', 0, 1), ('"one"', 1, 1), 
			('keyTwo', 0, 1), ('2', 1, 1)
		]

		expected_interpretation = {'keyOne': 'one', 'keyTwo': 2}
		got_interpretation = self.interp.interpret(tokens)

		self.assertEqual(got_interpretation, expected_interpretation)
		

	def test_nested_list_correct(self):
		tokens = [
			('keyOne', 0, 1), ('[', 2, 1), 
				('subKeyOne', 0, 2), ('"one"', 1, 2), 
				('subKeyTwo', 0, 3), ('[', 2, 3), 
					('subSubKey', 0, 4), ('"one"', 1, 4), 
					('subSubKey', 0, 5), ('2', 1, 5), 
				(']', 3, 6), 
			(']', 3, 7), 
			('keyTwo', 0, 8), ('"two"', 1, 8)
		]

		expected_interpretation = {
			'keyOne': {
				'subKeyOne': 'one',
				'subKeyTwo': {
					'subSubKey': ['one', 2]
				}
			}, 
			'keyTwo': 'two'
		}

		got_interpretation = self.interp.interpret(tokens)

		self.assertEqual(got_interpretation, expected_interpretation)


	def test_correct_empty_list(self):

		tokens =  [
			('keyOne', 0, 1), ('[', 2, 1), 
			(']', 3, 2), ('keyTwo', 0, 3), ('"two"', 1, 3)
		]

		expected_interpretation = {'keyOne': {}, 'keyTwo': 'two'}
		got_interpretation = self.interp.interpret(tokens)

		self.assertEqual(got_interpretation, expected_interpretation)


	def test_incorrect_val_when_key_expected(self):
		# VAL_TOK when KEY_TOK expected
		tokens = [
			('"keyOne"', 1, 1), ('"one"', 1, 1), 
			('keyTwo', 0, 1), ('2', 1, 1)
		]
		self.assertRaises(ZenException, self.interp.interpret, tokens)


	def test_incorrect_key_when_val_expected(self):
		# KEY_TOK when VAL_TOK expected
		tokens = [
			('keyOne', 1, 1), ('one', 0, 1), 
			('keyTwo', 0, 1), ('2', 1, 1)
		]
		self.assertRaises(ZenException, self.interp.interpret, tokens)


	def test_incorrect_unexpected_token_type(self):
		# unexpected token type
		tokens = [
			('keyOne', 1, 1), ('"one"', 4, 1), 
			('keyTwo', 0, 1), ('2', 1, 1)
		]
		self.assertRaises(ZenException, self.interp.interpret, tokens)


	def test_incorrect_eof_when_expecting_value(self):
		tokens = [
			('keyOne', 0, 1), ('"one"', 1, 1), 
			('keyTwo', 0, 1)
		]
		self.assertRaises(ZenException, self.interp.interpret, tokens)


	def test_incorrect_eolist_when_expecting_value(self):
		tokens = [
			('keyOne', 0, 1), ('[', 2, 1), 
				('subKeyOne', 0, 2), ('"one"', 1, 2), 
				('subKeyTwo', 0, 3),
			(']', 3, 6), 
			('keyTwo', 0, 8), ('"two"', 1, 8)
		]

		self.assertRaises(ZenException, self.interp.interpret, tokens)

	
if __name__ == '__main__':
	unittest.main()
