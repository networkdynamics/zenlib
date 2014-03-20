"""
This modules provides routines that concern reachability characteristics of 
a system and a control set.
"""

from zen import DiGraph, maximum_matching_
from zen.exceptions import type_check

def num_min_controls(G):
	"""
	Return the smallest number of controls that are required to control the graph ``G``
	assuming structural controllability conditions.
	"""
	type_check(G,DiGraph)
	
	matched_edges = None
	
	matched_edges = maximum_matching_(G)
	
	return max([len(G) - len(matched_edges),1])
	