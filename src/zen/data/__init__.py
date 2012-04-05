"""
This package contains various network data sets.  Convenience functions are provided for loading them.  The convenience functions load the data in
such a way that the node indices always correspond to the same nodes.
"""

import os
import os.path
import zen.io.scn as scn
import zen.io.gml as gml

__all__ = ['les_miserable','karate_club','florentine','axis_allies']

def les_miserable():
	"""
	Load and return an undirected network of the coappearances among the characters in the novel _Les Miserable_.
	
	Source: D. E. Knuth, The Stanford GraphBase: A Platform for Combinatorial Computing, Addison-Wesley, Reading, MA (1993).
	"""
	fname = os.path.join(os.path.dirname(__file__),'lesmis.gml')
	return gml.read(fname,weight_fxn=lambda x: x['value'])
	
def karate_club():
	"""
	Load and return an undirected social network of friendships among 34 members of a karate club in the 1970s.
	
	Source: W. W. Zachary, An information flow model for conflict and fission in small groups, Journal of Anthropological Research 33, 452-473 (1977).
	"""
	fname = os.path.join(os.path.dirname(__file__),'karate.scn')
	return scn.read(fname,directed=False)
	
def florentine():
	"""
	Load and return the undirected Florentine marriage network.
	
	Source: Padgett, John F. 1994. Marriage and Elite Structure in Renaissance Florence, 1282-1500. Paper delivered to the Social Science History Association.
	"""
	fname = os.path.join(os.path.dirname(__file__),'florentine.scn')
	return scn.read(fname,directed=False)
	
def axis_allies():
	pass