from zen import *
import unittest
import os
import os.path as path
import tempfile

class GMLTokenizerCase(unittest.TestCase):
		
	def test_basic_correct(self):
		gml_string = 'keyOne "one" keyTwo 2'
		tok = Tokenizer.Tokenizer()

		got_tokens = tok.tokenize(gml_string)
		expected_tokens = [('keyOne', 0, 0), ('"one"', 1, 0), ('keyTwo', 0, 0), ('2', 1, 0)]

		self.assertEqual(got_tokens, expected_tokens)
		

	def test_nested_list_correct(self):
		gml_string = '''
			keyOne [
				subKeyOne "one"
				subKeyTwo [
					subSubKey "one"
					subSubKey 2
				]
			]
			keyTwo "two"
		'''

		tok = Tokenizer.Tokenizer()

		got_tokens = tok.tokenize(gml_string)
		expected_tokens = [
			('keyOne', 0, 1), ('[', 2, 1), 
				('subKeyOne', 0, 2), ('"one"', 1, 2), 
				('subKeyTwo', 0, 3), ('[', 2, 3), 
					('subSubKey', 0, 4), ('"one"', 1, 4), 
					('subSubKey', 0, 5), ('2', 1, 5), 
				(']', 3, 6), 
			(']', 3, 7), 
			('keyTwo', 0, 8), ('"two"', 1, 8)
		]

		self.assertEqual(got_tokens, expected_tokens)

	def test_correct_with_comments(self):
		gml_string = '''
			# comment
			keyOne# comment
			"#one"# comment
			keyTwo # comment
			2 # comment
			#comment
		'''

		tok = Tokenizer.Tokenizer()

		got_tokens = tok.tokenize(gml_string)
		expected_tokens = [
			('keyOne', 0, 2), ('"#one"', 1, 3), 
			('keyTwo', 0, 4), ('2', 1, 5)
		]

		self.assertEqual(got_tokens, expected_tokens)

	def test_correct_empty_list(self):
		gml_string = '''
			keyOne [
			]
			keyTwo "two"
		'''

		tok = Tokenizer.Tokenizer()

		got_tokens = tok.tokenize(gml_string)
		expected_tokens =  [
			('keyOne', 0, 1), ('[', 2, 1), 
			(']', 3, 2), ('keyTwo', 0, 3), ('"two"', 1, 3)
		]

		self.assertEqual(got_tokens, expected_tokens)


	def test_incorrect_eof_in_string(self):
		gml_string = 'keyOne "one'
		tok = Tokenizer.Tokenizer()
		self.assertRaises(ZenException, tok.tokenize, gml_string)
		

	def test_incorrect_string_as_key(self):
		gml_string = 'keyOne "one" "keyTwo" 2'
		tok = Tokenizer.Tokenizer()
		self.assertRaises(ZenException, tok.tokenize, gml_string)

	def test_incorrect_eof_when_expecting_value(self):
		gml_string = 'keyOne "one" keyTwo'
		tok = Tokenizer.Tokenizer()
		self.assertRaises(ZenException, tok.tokenize, gml_string)


	def test_incorrect_eolist_when_expecting_value(self):
		gml_string = '''
			keyOne [
				subKeyOne "one"
				subKeyTwo 
			]
			keyTwo "two"
		'''
		tok = Tokenizer.Tokenizer()
		self.assertRaises(ZenException, tok.tokenize, gml_string)

	

#	def test_read_undirected_test1(self):
#		fname = path.join(path.dirname(__file__),'test2.gml')
#		G = gml.read(fname)
#		
#		self.assertEqual(len(G),3)
#		self.assertEqual(G.size(),2)
#		
#		self.assertEqual(type(G),Graph)
#		self.assertTrue(G.has_edge('N1','N2'))
#		self.assertTrue(G.has_edge('N2','N3'))
#		self.assertFalse(G.has_edge('N1','N3'))
#		self.assertTrue(G.has_edge('N3','N2'))
#		
#		self.assertEqual(G.node_idx('N1'),1)
#		self.assertEqual(G.node_idx('N2'),2)
#		self.assertEqual(G.node_idx('N3'),3)
#		
#		self.assertEqual(G.node_data('N1')['sample1'],42)
#		self.assertEqual(G.node_data('N2')['sample2'],42.1)
#		self.assertEqual(G.node_data('N3')['sample3'],'HELLO WORLD')
#		
#		self.assertEqual(G.edge_data('N1','N2')['label'],'Edge from node 1 to node 2')
#		
#	def test_list_variables(self):
#		fname = path.join(path.dirname(__file__),'test3.gml')
#		G = gml.read(fname)
#		
#		self.assertEqual(len(G),3)
#		self.assertEqual(G.size(),2)
#		
#		self.assertEqual(G.node_data('N1')['listVar'],[1,'a',3.2])
#		
#	def test_weight_fxn(self):
#		fname = path.join(path.dirname(__file__),'test3.gml')
#		G = gml.read(fname,weight_fxn=lambda data:data['value'])
#		
#		self.assertEqual(len(G),3)
#		self.assertEqual(G.size(),2)
#		
#		self.assertEqual(G.weight('N1','N2'),2)
#		self.assertEqual(G.weight('N2','N3'),3)
		
if __name__ == '__main__':
	unittest.main()
