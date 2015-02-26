"""
This module implements core control profile plotting functionality.  The module published two core functions that produce
profile plots and heatmaps.

.. autofunction:: profile_plot(items,...)

.. autofunction:: profile_heatmap(items,...)

.. seealso::

	J. Ruths and D. Ruths (2014). Control Profiles of Complex Networks. Science, 343(6177), 1373-1376.

"""

import math
import matplotlib
import matplotlib.pyplot as pyplot
from types import TupleType, ListType
from zen import DiGraph
from zen.control import profile as control_profile

__all__ = ['profile_plot','profile_heatmap','profile_heatmap_weighted']

## Constants ##

_SQRT3OVER2 = math.sqrt(3) / 2.

## Helpers ##
def _unzip(l):
	#return [x for (x,y) in l], [y for (x,y) in l]
	return zip(*l)

def _normalize(xs):
	s = float(sum(xs))
	return [x / s for x in xs]

## Boundary ##

def _draw_boundary(scale=1.0, linewidth=2.0, color='black'):
	# Plot boundary of 3-simplex.
	scale = float(scale)
	e = 0.03*scale
	# Note that the math.sqrt term is such to prevent noticable roundoff on the top corner point.
	pyplot.plot([0-e/1.25, scale+e/1.25, scale / 2, 0-e/1.25], [0-e/2.2, 0-e/2.2, math.sqrt(3.0*(scale+e)**2) / 2, 0-e/2.2], color, linewidth=linewidth)
	pyplot.ylim([-0.05 * scale, .90 * scale])
	pyplot.xlim([-0.05 * scale, 1.05 * scale])

## Curve Plotting ##
def _project_point(p):
	"""Maps (x,y,z) coordinates to planar-simplex."""
	a = p[0]
	b = p[1]
	c = p[2]
	x = 0.5 * (2 * b + c)
	y = _SQRT3OVER2 * c
	return (x, y)

def _project(s):
	"""Maps (x,y,z) coordinates to planar-simplex."""
	# Is s an appropriate sequence or just a single point?
	try:
		return _unzip(map(_project_point, s))
	except TypeError:
		return _project_point(s)
	except IndexError: # for numpy arrays
		return _project_point(s)

def _plot(t, color=None, marker=None, markersize=8.0, mfc='none', mec=None, mew=1.0, label=None, picker=None):
	"""Plots trajectory points where each point satisfies x + y + z = 1."""
	xs, ys = _project(t)
	if color and marker:
		if mfc == 'none' and mec is None:
			mec = color
		pyplot.plot(xs, ys, color=color, marker=marker, mew=mew, mec=mec, mfc=mfc, mfcalt='none', markersize=markersize, rasterized=False, label=label, picker=picker)
	else:
		pyplot.plot(xs, ys, linewidth=linewidth, label=label, picker=picker)

## Heatmaps##

def _simplex_points(steps=100, boundary=True):
	"""Systematically iterate through a lattice of points on the simplex."""
	steps = steps - 1
	start = 0
	if not boundary:
		start = 1
	for x1 in range(start, steps + (1-start)):
		for x2 in range(start, steps + (1-start) - x1):
			x3 = steps - x1 - x2
			yield (x1, x2, x3)

def _colormapper(x, a=0, b=1, cmap=None):
	"""Maps color values to [0,1] and obtains rgba from the given color map for triangle coloring."""
	if b - a == 0:
		rgba = cmap(0)
	else:
		rgba = cmap((x - a) / float(b - a))
	hex_ = matplotlib.colors.rgb2hex(rgba)
	return hex_

def _triangle_coordinates(i, j, steps):
	"""Returns the ordered coordinates of the triangle vertices for i + j + k = 1."""
	steps = float(steps)
	if j % 2 == 0: # j is even, triangle with point at top
		return [ ( (i+j+1)/steps/2., (i+1)/steps*_SQRT3OVER2 ), ( (i+j)/steps/2., i/steps*_SQRT3OVER2 ), ( (i+j+2)/steps/2., i/steps*_SQRT3OVER2 ) ]
	else: # j is odd, triangle with point at bottom
		return [ ( (i+j+1)/steps/2., i/steps*_SQRT3OVER2 ), ( (i+j)/steps/2., (i+1)/steps*_SQRT3OVER2 ), ( (i+j+2)/steps/2., (i+1)/steps*_SQRT3OVER2 ) ]

# def _triangle_coordinates(i, j, steps, alt=False):
# 	"""Returns the ordered coordinates of the triangle vertices for i + j + k = N. Alt refers to the averaged triangles; the ordinary triangles are those with base parallel to the axis on the lower end (rather than the upper end)"""
# 	steps = float(steps)
# 	# N = i + j + k
# 	if not alt:
# 		return [((i/2. + j)/steps, i * _SQRT3OVER2/steps), ((i/2. + j + 1)/steps, i * _SQRT3OVER2/steps), ((i/2. + j + 0.5)/steps, (i + 1) * _SQRT3OVER2/steps)]
# 	else:
# 		# Alt refers to the inner triangles not covered by the default case
# 		return [((i/2. + j + 1)/steps, i * _SQRT3OVER2/steps), ((i/2. + j + 1.5)/steps, (i + 1) * _SQRT3OVER2/steps), ((i/2. + j + 0.5)/steps, (i + 1) * _SQRT3OVER2/steps)]

def _heatmap(d, steps, cmap_name=None):
	"""Plots counts in the dictionary d as a heatmap. d is a dictionary of (i,j) --> c pairs where N = i + j + k."""
	if not cmap_name:
		cmap =  _white2colorcmap('#0000FF')
	else:
		cmap = pyplot.get_cmap(cmap_name)
	# Colorbar hack -- make fake figure and throw it away.
	#Z = [[0,0],[0,0]]
	#levels = [v for v in d.values()]
	#levels.sort()
	#CS3 = pyplot.contourf(Z, levels, cmap=cmap)
	# Plot polygons
	#pyplot.clf()
	a = min(d.values())
	b = max(d.values())
	# Color data triangles.
	for k, v in d.items():
		i, j = k
		vertices = _triangle_coordinates(i,j,steps)
		x,y = _unzip(vertices)
		color = _colormapper(d[i,j],a,b,cmap=cmap)
		pyplot.fill(x, y, facecolor=color, edgecolor='none')
	# Color smoothing triangles.
	#for i in range(steps+1):
	#	for j in range(steps - i):
	#		try:
	#			alt_color = (d[i,j] + d[i, j + 1] + d[i + 1, j])/3.
	#			color = _colormapper(alt_color, a, b, cmap=cmap)
	#			vertices = _triangle_coordinates(i,j,steps, alt=True)
	#			x,y = _unzip(vertices)
	#			pyplot.fill(x, y, facecolor=color, edgecolor=color)
	#		except KeyError:
	#			# Allow for some portions to have no color, such as the boundary
	#			pass
	#Colorbar hack continued.
	#pyplot.colorbar(CS3)

def _white2colorcmap(arg):
	CC = matplotlib.colors.ColorConverter()
	r,g,b = CC.to_rgb(arg)
	cmap = { 'red': ((0.0,1.0,1.0),(1.0,r,r)),
			  'green': ((0.0,1.0,1.0),(1.0,g,g)),
			  'blue': ((0.0,1.0,1.0),(1.0,b,b))}
	return matplotlib.colors.LinearSegmentedColormap('mycustom',cmap,256)

def _heatmap_scatter(pts, steps):
	d = dict()
	for i in range(steps):
		for j in range(2*(steps-i)-1):
			d[(i,j)] = 0
	
	for (x,y,z) in pts:
		# find projection into the simplex
		(xt,yt) = _project((x,y,z))
		# print '(%1.2f, %1.2f, %1.2f) projects to (%1.2f, %1.2f)' % (x,y,z,xt,yt)
		# vertical index
		i = math.trunc((yt/_SQRT3OVER2)*steps)
		# the very tip of the triangle should go in the top subtriangle
		if i == steps:
			i = steps-1
		# print 'vertical index i = %i' % i
		
		if i == steps-1:
			j = 0
		else:
			# horizontal index
			k = math.trunc((xt - float(i)*(1./steps/2.))*steps*2) # 'divide' by half the width of a subtriangle
			# print 'horizontal half widths: k = %i' % k
			if k == 0:
				j = k
			elif k >= 2*(steps-i)-1:
				j = 2*(steps-i)-2
			else:
				xtt = ( xt - float(k)*(1./steps/2.) )/(1./steps/2.) # find the part of xt to the right of the k-th triangle half-width, normalized
				ytt = ( yt - float(i)*(_SQRT3OVER2/steps) )/(_SQRT3OVER2/steps)  # find height within the i-th row, normalized
				# print 'point in normalized area: (%1.2f, %1.2f)' % (xtt,ytt)
				if k % 2 == 1:  # k is odd, so triangle boundary goes down-right
					if ytt <= -xtt + 1: # point is left of the triangle boundary
						j = k-1
					else: # point is right of the triangle boundary
						j = k
				else:  # k is even, so triangle boundary goes up-right
					if ytt <= xtt: # point is left of the triangle boundary
						j = k-1
					else: # point is right of the triangle boundary
						j = k
		# print 'dictionary index: (%i, %i)' % (i,j)
		d[(i,j)] += 1
	
	return d

def profile_heatmap(items, **kwargs):
	"""
	Plots a set of control profiles as a heatmap on a triangular control profile plot. Each of the ``items`` specified
	can be either a control profile 3-tuple, 3-list, or :py:class:`zen.DiGraph` (in which case the control profile of the
	graph will be computed and then plotted).
	
	The resolution of the mesh can be controlled using ``num_steps``. If no matplotlib color map name or color map is supplied (``cmap``), one will be
	generated using a gradient between white and color.
	
	**KwArgs**:
	
		* ``num_steps [=15]`` (``int``). The resolution of the heatmap mesh.
		* ``cmap [=None]`` (``colormap``). The colormap that will be used when producing the heatmap.
		* ``color [='b']`` (any matplotlib-supported color). Specify a color instead of a colormap for ease.
	
	**Returns**:
		A dictionary with boundaries for the individual regions being rendered in the heatmap.
		
	"""
	######
	# Handle relevant arguments
	if type(items) is not ListType:
		items = [items]
		
	num_steps = kwargs.pop('num_steps',15)
	cmap = kwargs.pop('cmap',None)
	color = kwargs.pop('color','b')
	
	return _plot_profiles(	items,
							heatmap = True,
							num_steps = num_steps,
							cmap = cmap,
							color = color)
	
def profile_plot(items,**kwargs):
	"""
	Plots a set of control profiles on a triangular control profile plot. Each of the ``items`` specified
	can be either a control profile 3-tuple, 3-list, or :py:class:`zen.DiGraph` (in which case the control profile of the
	graph will be computed and then plotted).
	
	Most of the usual matplotlib plotting features are supported (see keyword arguments supported below).
	
	**KwArgs**:
	
		* ``color [='b']`` (any matplotlib-supported color). The color of the markers.
		* ``marker [='o']`` (any matplotlib-supported marker). The marker that will be used for each control profile.
		* ``markersize [=8.0]`` (``float``).  The size of the marker.
		* ``mfc [='none']`` (any matplotlib-supported color). The color of the marker face.
		* ``mec [=None]`` (any matplotlib-supported color). The color of the marker edge face.
		* ``mew [=1.0]`` (``float``). The weight of the marker edge.
		
	"""
	######
	# Handle relevant arguments
	if type(items) is not ListType:
		items = [items]
			
	color = kwargs.pop('color','b')
	marker = kwargs.pop('marker','o')
	markersize = kwargs.pop('markersize',8.0)
	mfc = kwargs.pop('mfc','none')
	mec = kwargs.pop('mec',None)
	mew = kwargs.pop('mew',1.0)
	
	# currently undocumented features
	label = kwargs.pop('label',None)
	picker = kwargs.pop('picker',None)
	
	# heatmap keywords we don't support
	heatmap = False
	num_steps = None # this is for the heatmap, so it doesn't matter
	cmap = None
	
	return _plot_profiles(	items,
							color = color,
							marker = marker,
							markersize = markersize,
							mfc = mfc,
							mec = mec,
							mew = mew,
							label = label,
							picker = picker,
							heatmap = heatmap,
							num_steps = num_steps,
							cmap = cmap)
		
def _plot_profiles(items, **kwargs): # heatmap=False, color='b', marker='o', markersize=8.0, mfc='none', mec=None, mew=1.0, label=None, picker=None, num_steps=15, cmap=None):
	"""
	This is the internal function for creating profile plots and heatmaps.  The reason for separating this out 
	is simply to clarify the arguments that are particular to plots vs. heatmaps.
	
	Plots the control profiles on a ternary plot. Specific points can be passed as 2- or 3-tuples or 2- or 3-lists.
	The points should obey x+y+z=1. Alternately, a DiGraph can be passed and the points will be calculated.
	Lists of these same components (or mixtures) can be accepted. Most of the usual matplotlib plotting features are 
	supported:   [color, marker, markersize, mfc, mec, mew, label, picker]
	
	If the heatmap=True flag is set, a heatmap will be generated instead of a scatter plot. The resolution of the mesh
	can be controlled using num_steps. If no matplotlib color map name or color map is supplied (cmap), one will be
	generated using a gradient between white and color.
	
	The arguments are documented in more detail below in comments.
	"""
	#####
	# handle arguments
	if type(items) is not ListType:
		items = [items]
		
	## general arguments
	# switch heatmap on or off
	heatmap = kwargs.pop('heatmap',False)
	
	# label associated with individual components.  Not documented right now.  Per Justin's comments:
	# [3/20/14, 2:37:21 PM] Justin Ruths: this is the field where I stored the name you would show
	# [3/20/14, 2:37:36 PM] Justin Ruths: it is probably not necessary, but if you wanted a legend, it would be handy
	label = kwargs.pop('label',None)
	
	# picker for picking points - also not documented right now
	picker = kwargs.pop('picker',None)
	
	## Plot only
	# set the plotting color (plot only)
	color = kwargs.pop('color','b')
	
	# marker shape (plot only)
	marker = kwargs.pop('marker','o')
	
	# marker size (plot only)
	markersize = kwargs.pop('markersize',8.0)
	
	# marker face color (plot only)
	mfc = kwargs.pop('mfc','none')
	
	# marker edge color (plot only)
	mec = kwargs.pop('mec',None)
	
	# marker edge width (plot only)
	mew = kwargs.pop('mew',1.0)
	
	## Heatmap only
	# number of steps (heatmap only)
	num_steps = kwargs.pop('num_steps',15)
	
	# color map (heatmap only)
	cmap = kwargs.pop('cmap',None)

	# actually make heatmap plot or not
	plot_heatmap = kwargs.pop('plot_heatmap',True)
	
	#####
	# Collect the control profiles to plot
	pts = []
	for item in items:
		if type(item) is TupleType or type(item) is ListType:
			p = tuple(item)
			if len(p) == 2:
				pts.append( (p[0],p[1],1-p[0]-p[1]) )
			elif len(p) == 3:
				pts.append( p )
			else:
				raise TypeError, 'plot_profiles supports lists of tupes/lists of length 2 or 3 only; found length %i.' % len(p)
		elif type(item) is DiGraph:
			pts.append(control_profile(item,normalized=True))
		else:
			raise TypeError, 'items of type %s cannot be converted into control profiles' % str(type(item))
	
	if heatmap:
		if cmap is None:
			cmap = _white2colorcmap(color)
		d = _heatmap_scatter(pts,num_steps)
		
		if plot_heatmap:
			_heatmap(d,num_steps,cmap)
			pyplot.axis('equal')
			pyplot.xticks([])
			pyplot.yticks([])
			_draw_boundary()
			pyplot.gca().set_frame_on(False)
		return d
	else:
		for p in pts:
			_plot(p, color=color, marker=marker, markersize=markersize, mfc=mfc, mec=mec, mew=mew, label=label, picker=picker)
		
		pyplot.axis('equal')
		pyplot.xticks([])
		pyplot.yticks([])
		_draw_boundary()
		pyplot.gca().set_frame_on(False)
		return None

def profile_heatmap_weighted(items, weights=None, **kwargs):
	"""
	Plots a weighted combination of control profiles as a heatmap on a triangular control profile plot. ``items`` is a list 
	of other lists composed of: control profile 3-tuple, 3-list, or :py:class:`zen.DiGraph` (in which case the control 
	profile of the graph will be computed and then plotted). A ``weights'' parameter can be specified to weight the 
	combination.
	
	The resolution of the mesh can be controlled using ``num_steps``. If no matplotlib color map name or color map is 
	supplied (``cmap``), one will be generated using a gradient between white and color.
	
	**KwArgs**:
	
		* ``num_steps [=15]`` (``int``). The resolution of the heatmap mesh.
		* ``cmap [=None]`` (``colormap``). The colormap that will be used when producing the heatmap.
		* ``color [='b']`` (any matplotlib-supported color). Specify a color instead of a colormap for ease.
		
	**Returns**:
		A dictionary with boundaries for the individual regions being rendered in the heatmap.
		
	"""
	######
	# Handle relevant arguments
	if type(items) is not ListType:
		items = [items]
		
	num_steps = kwargs.pop('num_steps',15)
	cmap = kwargs.pop('cmap',None)
	color = kwargs.pop('color','b')

	D = None
	if weights is None: # if no weights are given make them weighted equally
		weights = [1.0/float(len(items)) for i in range(len(items))]
	if len(weights) != len(items):
		raise Exception, 'weights parameter must be the same length as items.'
	
	for i,item in enumerate(items):
		d = _plot_profiles(item, heatmap=True, color=color, num_steps=num_steps, cmap=cmap, plot_heatmap=False)
		if D is None:
			D = dict()
			for point in d.keys():
				D[point] = 0
		for point, val in d.items():
			D[point] += float(val)*weights[i]/float(sum(weights))
	
	if cmap is None:
		cmap = _white2colorcmap(color)
	
	_heatmap(D, num_steps, cmap)
	pyplot.axis('equal')
	pyplot.xticks([])
	pyplot.yticks([])
	_draw_boundary()
	pyplot.gca().set_frame_on(False)
	return D
