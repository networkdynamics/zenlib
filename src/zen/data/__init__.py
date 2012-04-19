"""
The ``zen.data`` package bundles a number of public, frequently-used network datasets into the Zen library.  The intent of including these is to both make it easier for new users to play with real data and to enable established users to rapidly test ideas (algorithms, analysis, etc...) on known datasets.

While the ``zen.data`` package contains the data files for the various datasets, convenience functions are provided for loading each of the different datasets that are included.  Presently, the following datasets are included:

	* Knuth's *Les Miserable* dataset (``lesmis.gml``)
	* Padgett's *Florentine marriage* network (``florentine.scn``)
	* Zachary's Karate club network (``karate.scn``)
	
"""

import os
import os.path
import zen.io.scn as scn
import zen.io.gml as gml

__all__ = ['les_miserable','karate_club','florentine','axis_allies']

def les_miserable():
	"""
	Loads and returns an undirected network of the coappearances among the characters in the novel :title:`Les Miserable`.
	
	.. note::
		Source: D. E. Knuth, :title:`The Stanford GraphBase: A Platform for Combinatorial Computing`, Addison-Wesley, Reading, MA (1993).
	"""
	fname = os.path.join(os.path.dirname(__file__),'lesmis.gml')
	return gml.read(fname,weight_fxn=lambda x: x['value'])
	
def karate_club():
	"""
	Loads and returns an undirected social network of friendships among 34 members of a karate club in the 1970s.
	
	.. note::
		Source: W. W. Zachary, *An information flow model for conflict and fission in small groups*, Journal of Anthropological Research 33, 452-473 (1977).
	"""
	fname = os.path.join(os.path.dirname(__file__),'karate.scn')
	return scn.read(fname,directed=False)
	
def florentine():
	"""
	Loads and returns the undirected Florentine marriage network.
	
	.. note::
		Source: Padgett, John F. 1994. *Marriage and Elite Structure in Renaissance Florence*, 1282-1500. Paper delivered to the Social Science History Association.
	"""
	fname = os.path.join(os.path.dirname(__file__),'florentine.scn')
	return scn.read(fname,directed=False)
	
def axis_allies():
	pass