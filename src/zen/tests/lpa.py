import unittest
import os, os.path

from zen.algorithms.community import lpa
import zen

class RealNetworksLPATestCase(unittest.TestCase):

	def test_mondego(self):
		
		# load the mondego network
		net_fname = os.path.join(os.path.dirname(__file__),'mondego.edgelist')

		G = zen.io.edgelist.read(net_fname,directed=True)

		# find the communities
		comms = lpa(G)

		# verify that all nodes are in the communities
		nodes = set(G.nodes())

		print 'NET:',sorted(G.nodes(),cmp=lambda x,y: cmp(int(x),int(y)))
		
		for c in comms:
			print sorted(c.nodes(),cmp=lambda x,y: cmp(int(x),int(y)))
			print sorted(c.nodes_())
			for n in c:
				if n not in nodes:
					self.fail('node was not found in node set: %s' % str(n))
				nodes.remove(n)

		# see if there are any others
		self.assertEqual(len(nodes),0,'not all nodes were in communities: %s' % str(nodes))

