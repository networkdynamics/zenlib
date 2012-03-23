"""
This subpackage contains graph layout algorithms.  All graph layout functions produce a new or modify an existing view object.
Graph layout functions all return the view object produced or modified.

In addition to algorithm-specific parameters, each layout function accepts one positional argument:

	- a graph or view object (required as the first argument).  If a graph, then a new view is created.
	
All layout functions also accept the following keyword arguments:

	- bbox is a tuple (left,bottom,top,right) indicating the bounds that should be applied to the layout. If set to None,
	  then node locations are unconstrained.  By default, the bounding box is (0,0,100,100).
	
"""

import spring_layout
import random_layout
import forceatlas_layout
import fruchtermanreingold_layout

BBOX_NAME = 'bbox'
BBOX_DEFAULT_VALUE = (0.,0.,100.,100.)

def spring(GV,**kwargs):
	if BBOX_NAME not in kwargs:
		kwargs[BBOX_NAME] = BBOX_DEFAULT_VALUE
		
	return spring_layout.layout(GV,**kwargs)
	
def random(GV,**kwargs):
	if BBOX_NAME not in kwargs:
		kwargs[BBOX_NAME] = BBOX_DEFAULT_VALUE
		
	return random_layout.layout(GV,**kwargs)
	
def forceatlas(GV,**kwargs):
	if BBOX_NAME not in kwargs:
		kwargs[BBOX_NAME] = BBOX_DEFAULT_VALUE

	return forceatlas_layout.layout(GV,**kwargs)
	
def fruchtermanreingold(GV,**kwargs):
	if BBOX_NAME not in kwargs:
		kwargs[BBOX_NAME] = BBOX_DEFAULT_VALUE

	return fruchtermanreingold_layout.layout(GV,**kwargs)