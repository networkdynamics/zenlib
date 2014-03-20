"""
This module implements functions surrounding the calculation of control profiles for networks.

.. seealso::

	J. Ruths and D. Ruths (2014). Control Profiles of Complex Networks. Science, 343(6177), 1373-1376.

"""
from zen.digraph import DiGraph
from zen.exceptions import type_check

from reachability import num_min_controls

def profile(G,**kwargs):
	"""
	Compute the control profile for the directed network ``G``.
	
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
	normalized = kwargs.pop('normalized',True)
	type_check(normalized,bool)
	
	Nc = float(num_min_controls(G))
	
	# source dilations
	Ns = float(G.num_sources)
	
	# external dilations
	Nsink = float(G.num_sinks)
	Ne = max([0.0,Nsink - Ns])
	
	# internal dilations
	if Nc == 1 and Ns == 0 and Nsink == 0:
		# this condition handles the case where the network consists of one or more cycles
		# thereby requiring only one control to drive all of them.
		Ni = 0
	else:
		Ni = Nc - Ns - Ne
	
	if normalized:
		return (Ns/Nc, Ne/Nc, Ni/Nc)
	else:
		return (Ns,Ne,Ni)
