from zen.view import View
__all__ = ['layout']
import numpy as np
import math
import random

X = 0
Y = 1

cpdef layout(GV,bbox):
		
	#find graph.view:
	if type(GV) == View:
		view = GV
		graph = view.graph()
	else:
		graph = GV
		view = View(graph)

	#call forces as many times as necessary
	conv_threshold = 0.008 #smaller value means better network but more time (too small is bad)
	#count = 0
	while 1:
		pos_array, dx, dy = run_forces(view)
		if np.sum(dx)==0 and np.sum(dy)==0:
			break
		view.set_pos_array(pos_array)
		#print np.average(np.absolute(dx))
		#print np.average(np.absolute(dy))
		if np.average(np.absolute(dx)) <= conv_threshold and np.average(np.absolute(dy)) <= conv_threshold :
			break
		#count += 1
	view.set_pos_array(normalize(pos_array,graph))
	return view
		
def run_forces(view):
	
	#get position array:
	graph = view.graph()
	pos_array = view.pos_array()
	if pos_array == None:
		pos_array = rand_pos_array(view)
		view.set_pos_array(pos_array)
	
	#get a blank dx dy array
	(dx,dy) = newdd(len(graph))
	
	dx, dy = repulsion(view, dx, dy)
	dx, dy = attraction(view, dx, dy)
	
	#apply changes and return result
	pos_array = apply_dd(graph,pos_array,dx,dy)
	return (pos_array,dx,dy)

#Methods:

def repulsion(view,dx,dy):
	graph = view.graph()
	pos_array = view.pos_array()
	
	for n in graph.nodes_iter_():
		for nn in graph.nodes_iter_():
			if n != nn:
				(dx,dy) = repel(n,nn,dx,dy,view)
				
	return (dx,dy)
	
def attraction(view,dx,dy):
	graph = view.graph()
	pos_array = view.pos_array()
	
	for e in graph.edges_():
		n = graph.endpoints_(e)[0]
		nn = graph.endpoints_(e)[1]
		if n != nn:
			(dx,dy) = attract(n,nn,dx,dy,view) 	
				
	return (dx,dy)
	
def repel(n,nn,dx,dy,view):
	graph = view.graph()
	pos_array = view.pos_array()
	(xDist,yDist,dist) = node_dist(n,nn,pos_array)
	rc = repulsion_constant(dist)

	if dist > 0:
	
		dx[n] += rc * xDist
		dy[n] += rc * yDist 
	
		dx[nn] -= rc * xDist
		dy[nn] -= rc * yDist
		
		return (dx,dy)
		
def attract(n,nn,dx,dy,view):
	graph = view.graph()
	pos_array = view.pos_array()
	(xDist,yDist,dist) = node_dist(n,nn,pos_array)
	ac = attraction_constant(dist)

	if dist > 0:
	
		dx[n] += ac * xDist
		dy[n] += ac * yDist 
	
		dx[nn] -= ac * xDist
		dy[nn] -= ac * yDist
		
		return (dx,dy)

def repulsion_constant(dist):
	rc = 0.003 / (dist*dist) #larger value means one step does more [there is a delicate balance required here]
	return rc
	
def attraction_constant(dist):
	ac = -0.03 #larger absolute value means one step does more
	return ac

def node_dist(n,nn,pos_array):
	xDist = pos_array[n,X] - pos_array[nn,X]
	yDist = pos_array[n,Y] - pos_array[nn,Y]
	dist =  math.sqrt(xDist * xDist + yDist * yDist)
	return (xDist,yDist,dist)
	
def apply_dd(graph,pos_array,dx,dy):
	for n in graph.nodes_iter_():
		pos_array[n,X] += dx[n]
		pos_array[n,Y] += dy[n]
	return pos_array

def newdd(n):
	dx = []
	dy = []
	for n in range(0,n):
		dx.append(0)
		dy.append(0)
	return (dx, dy)
	
def rand_pos_array(view):
	graph = view.graph()
	pos_array = np.zeros( (graph.max_node_idx+1,2), np.float)
	for n in graph.nodes_iter_():
		pos_array[n,X] = (random.random())
		pos_array[n,Y] = (random.random())
		#for testing:
		#pos_array[n,X] = 0.5 + 0.001*n
		#pos_array[n,Y] = 0.5 - 0.001*n
	return pos_array
	
def normalize(pa,graph):
	if np.sum(pa) > 0:
		box = (5,5)
	
		desiredxrange = box[X]
		desiredyrange = box[Y]
		xmax = np.max(pa[:,X])
		xmin = np.min(pa[:,X])
		ymax = np.max(pa[:,Y])
		ymin = np.min(pa[:,Y])

		if xmin < 0:
			for n in graph.nodes_iter_():
				pa[n,X] += abs(xmin)
		if ymin < 0:
			for n in graph.nodes_iter_():
				pa[n,Y] += abs(ymin)
		xrange = xmax - xmin
		yrange = ymax - ymin
		xratio = desiredxrange / xrange
		yratio = desiredyrange / yrange

		for n in graph.nodes_iter_():
			pa[n,X] *= xratio
			pa[n,Y] *= yratio
	return pa