from zen.view import View
import numpy as np
import random

	
__all__ = ['layout']

cpdef layout(GV,bbox=(0,0,100,100) ):
	"""
	TODO(Ben): Implement the bounding box that influences where the random nodes are placed.
	"""
	view = None
	graph = None

	if type(GV) == View:
		view = GV
		graph = view.graph()
	else:
		graph = GV
		view = View(graph)
		
	# initialize the 
	pos_array = np.zeros( (graph.max_node_idx+1,2), np.float)
	
	for n in graph.nodes_iter_():
		pos_array[n,0] = (random.random()) * 100.
		pos_array[n,1] = (random.random()) * 100.
	
	view.set_pos_array(pos_array)
	
	return view