"""
This module implements functions surrounding the calculation of control profiles for networks.
"""
from zen.digraph import DiGraph

def profile(G,**kwargs):
	"""
	Compute the control profile for the directed network ``G``.  THe 
	
	**KwArgs**:
	  	* ``normalized [=True]`` (``boolean``). Indicates whether each element in the control
		  profile should be normalized by the number of controls.
	
	**Returns**:
		``(s,e,i)``. The control profile consisting of the source controls, external-dilation controls,
		and internal-dilation controls.  Un-normalized, these values with the the number of each.  Normalized
		these values will be the fraction of all controls needed by the network belonging to each type.
	"""
	# this only works on directed graphs
	type_check(G,DiGraph,'Only directed graphs are supported')

	# load keyword arguments
	normalized = kwargs.pop('normalize',True)
	type_check(normalized,bool)
	
	Nc = float(generic_rank(G))
	Ns = float(G.num_sources)
	Nsink = float(G.num_sinks)
	Ne = max([0.0,Nsink - Ns])
	Ni = Nc - Ns - Ne
	
	if normalized:
		return (Ns/Nc, Ne/Nc, Ni/Nc)
	else:
		return (Ns,Ne,Ni)
