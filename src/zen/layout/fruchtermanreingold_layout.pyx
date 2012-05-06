from zen.view import View

__all__ = ['layout']

from zen.graph cimport Graph
from zen.digraph cimport DiGraph
import numpy as np
cimport numpy as np
from zen.exceptions import *
from zen.view import View
cimport cython
from libc.stdlib cimport RAND_MAX, rand

from cpython cimport bool

__all__ = ['layout']

cdef extern from "math.h":
	double sqrt(double x)
	
cdef inline float fast_max(float a, float b): return a if a >= b else b
cdef inline float fast_min(float a, float b): return a if a <= b else b

cpdef layout(GV,num_iterations=100,bbox=(0.,0.,100.,100.)):
	"""
	Use spring force layout to layout the nodes in graph (or view) GV.

	The result is a view containing the layout of this graph.  If a view
	was specified, then the resulting view will simply be this view object
	with the position array updated.
	
	The algorithm used is the Fruchterman and Reingold algorithm published in 1991
	in a paper entitled "Graph Drawing by Force-Directed Placement".
	"""	
	view = None
	G = None

	if type(GV) == View:
		view = GV
		G = view.graph()
	else:
		G = GV
		view = View(G)
		
	if type(G) == Graph:
		view.set_pos_array(undirected_spring(<Graph> G,num_iterations,bbox))
	elif type(G) == DiGraph:
		view.set_pos_array(directed_spring(<DiGraph> G,num_iterations,bbox))
	else:
		raise ZenException, 'Unsupported graph type: %s' % type(G)
		
	return view

cdef int repulsion_force(float ksq,float p0x,float p0y,float p1x,float p1y,float* fx,float* fy):
	cdef float dx = p0x - p1x
	cdef float dy = p0y - p1y
	cdef float dmag = sqrt(dx*dx + dy*dy)
	cdef float fr
	if dmag == 0:
		# the points are in the same place! randomize one of them
		# pos[nidx2,XP] = np.random.random() * width
		# pos[nidx2,YP] = np.random.random() * height
		return 1
	else:
		fr = ksq / (dmag*dmag)
		dx = (dx / dmag) * fr
		dy = (dy / dmag) * fr
		fx[0] = dx
		fy[0] = dy
		
	return 0

@cython.boundscheck(False) # turn off bounds checking for this function
cpdef undirected_spring(Graph G,int num_iterations,bounds):
	cdef float low_x = bounds[0]
	cdef float low_y = bounds[1]
	cdef float width = bounds[2] - bounds[0]
	cdef float height = bounds[3] - bounds[1]
	cdef float area = width * height
	cdef int nnodes = len(G)
	cdef int nnode_pos = G.next_node_idx
	cdef float k = sqrt(area/nnodes)
	cdef float ksq = k*k
	cdef int i
	cdef unsigned int nidx,eidx,nidx2
	cdef float center_x = width/2. + low_x
	cdef float center_y = height/2. + low_y
	cdef float fx
	cdef float fy
	
	cdef float temp = 0.1 * area #width * 0.1 # adjustments of at most one tenth of the frame
	#cdef float cool_frac = 0.95
	cdef float cool_amt = temp / num_iterations
		
	cdef float dx,dy,dmag,fr,fa
	cdef np.ndarray[np.float_t, ndim=2] pos = np.random.random( (nnode_pos,2) )
	cdef np.ndarray[np.float_t, ndim=2] disp = np.zeros( (nnode_pos,2), np.float)
	cdef unsigned int XP,YP
	XP = 0
	YP = 1
	
	# assign random positions
	pos[:,XP] *= width
	pos[:,XP] += low_x
	pos[:,YP] *= height
	pos[:,YP] += low_y
		
	# iterate through and improve positions
	for i in range(num_iterations):
		
		# calculate repulsive forces
		for nidx in range(G.next_node_idx):
			# skip invalid positions in the node info array
			if not G.node_info[nidx].exists:
				continue
			
			#####
			# walls exert repulsion
			
			# upper wall
			if repulsion_force(ksq,pos[nidx,XP],pos[nidx,YP],pos[nidx,XP],0,&fx,&fy) == 0:
				disp[nidx,XP] += fx
				disp[nidx,YP] += fy
			else:
				pos[nidx,XP] = (<float>rand()/(<float>RAND_MAX+1.)) * width
				pos[nidx,YP] = (<float>rand()/(<float>RAND_MAX+1.)) * height
				
			# lower wall
			if repulsion_force(ksq,pos[nidx,XP],pos[nidx,YP],pos[nidx,XP],height,&fx,&fy) == 0:
				disp[nidx,XP] += fx
				disp[nidx,YP] += fy
			else:
				pos[nidx,XP] = (<float>rand()/(<float>RAND_MAX+1.)) * width
				pos[nidx,YP] = (<float>rand()/(<float>RAND_MAX+1.)) * height
				
			# left wall
			if repulsion_force(ksq,pos[nidx,XP],pos[nidx,YP],0,pos[nidx,YP],&fx,&fy) == 0:
				disp[nidx,XP] += fx
				disp[nidx,YP] += fy
			else:
				pos[nidx,XP] = (<float>rand()/(<float>RAND_MAX+1.)) * width
				pos[nidx,YP] = (<float>rand()/(<float>RAND_MAX+1.)) * height

			# lower wall
			if repulsion_force(ksq,pos[nidx,XP],pos[nidx,YP],width,pos[nidx,YP],&fx,&fy) == 0:
				disp[nidx,XP] += fx
				disp[nidx,YP] += fy
			else:
				pos[nidx,XP] = (<float>rand()/(<float>RAND_MAX+1.)) * width
				pos[nidx,YP] = (<float>rand()/(<float>RAND_MAX+1.)) * height
			
			# add attraction towards the center
			# dx = pos[nidx,XP] - center_x
			# dy = pos[nidx,YP] - center_y
			# dmag = sqrt(dx*dx + dy*dy)
			# 
			# fa =  (dmag*dmag) / k
			# disp[nidx,XP] -= (dx/dmag) * fa
			# disp[nidx,YP] -= (dy/dmag) * fa
			
			for nidx2 in range(nidx+1,G.next_node_idx):
				# skip invalid positions in the node info array
				if not G.node_info[nidx2].exists:
					continue
				
				if repulsion_force(ksq,pos[nidx,XP],pos[nidx,YP],pos[nidx2,XP],pos[nidx2,YP],&fx,&fy) == 0:
					disp[nidx,XP] += fx
					disp[nidx,YP] += fy
					
					disp[nidx2,XP] -= fx
					disp[nidx2,YP] -= fy
				else:
					pos[nidx2,XP] = (<float>rand()/(<float>RAND_MAX+1.)) * width
					pos[nidx2,YP] = (<float>rand()/(<float>RAND_MAX+1.)) * height					
				
				# dx = pos[nidx,XP] - pos[nidx2,XP]
				# dy = pos[nidx,YP] - pos[nidx2,YP]
				# dmag = sqrt(dx*dx + dy*dy)
				# if dmag == 0:
				# 	# the points are in the same place! randomize one of them
				# 	# pos[nidx2,XP] = np.random.random() * width
				# 	# pos[nidx2,YP] = np.random.random() * height
				# 	pos[nidx2,XP] = (<float>rand()/(<float>RAND_MAX+1.)) * width
				# 	pos[nidx2,YP] = (<float>rand()/(<float>RAND_MAX+1.)) * height
				# else:
				# 	fr = ksq / (dmag*dmag)
				# 	dx = (dx / dmag) * fr
				# 	dy = (dy / dmag) * fr
				# 	disp[nidx,XP] += dx
				# 	disp[nidx,YP] += dy
				# 	
				# 	disp[nidx2,XP] -= dx
				# 	disp[nidx2,YP] -= dy
	
		# calculate attractive forces
		for eidx in range(G.next_edge_idx):
			# skip invalid positions in the edge info array
			if not G.edge_info[eidx].exists or G.edge_info[eidx].u == G.edge_info[eidx].v:
				continue
				
			dx = pos[G.edge_info[eidx].v,XP] - pos[G.edge_info[eidx].u,XP]
			dy = pos[G.edge_info[eidx].v,YP] - pos[G.edge_info[eidx].u,YP]
			dmag = sqrt(dx*dx + dy*dy)
			
			fa =  (dmag*dmag) / k
			disp[<unsigned int>G.edge_info[eidx].v,XP] -= (dx/dmag) * fa
			disp[<unsigned int>G.edge_info[eidx].v,YP] -= (dy/dmag) * fa
			disp[<unsigned int>G.edge_info[eidx].u,XP] += (dx/dmag) * fa
			disp[<unsigned int>G.edge_info[eidx].u,YP] += (dy/dmag) * fa
			
		# reposition nodes
		for nidx in range(G.next_node_idx):
			# skip invalid positions in the node info array
			if not G.node_info[nidx].exists:
				continue
			
			dx = disp[nidx,XP]
			dy = disp[nidx,YP]
			dmag = sqrt(dx*dx + dy*dy)
			pos[nidx,XP] += (dx/dmag) * fast_min(dmag,temp)
			pos[nidx,YP] += (dy/dmag) * fast_min(dmag,temp)
			
			# enforce bounds
			pos[nidx,XP] = fast_min(width, fast_max(0,pos[nidx,XP]))
			pos[nidx,YP] = fast_min(height, fast_max(0,pos[nidx,YP]))
			
			disp[nidx,XP] = 0
			disp[nidx,YP] = 0

		# reduce temperature
		temp = temp - cool_amt
	
	# update the positions to be within the box
	for nidx in range(G.next_node_idx):
		# skip invalid positions in the node info array
		if not G.node_info[nidx].exists:
			continue
			
		pos[nidx,XP] += low_x
		pos[nidx,YP] += low_y
	
	return pos
	
	