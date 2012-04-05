"""
This package contains various network data sets.  Convenience functions are provided for loading them.
"""

import os
import os.path
import zen.io.scn as scn

__all__ = ['les_miserable','karate_club','florentine','axis_allies']

def les_miserable():
	pass
	
def karate_club():
	pass
	
def florentine():
	"""
	Load and return the undirected Florentine marriage network.
	
	Source: Padgett, John F. 1994. Marriage and Elite Structure in Renaissance Florence, 1282-1500. Paper delivered to the Social Science History Association.
	"""
	fname = os.path.join(os.path.dirname(__file__),'florentine.scn')
	return scn.read(fname,directed=False)
	
def axis_allies():
	pass