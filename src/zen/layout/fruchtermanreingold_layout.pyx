from zen.view import View

__all__ = ['layout']

cpdef layout(GV,bbox):
	"""
	<DOC>
	"""
	view = None
	graph = None

	if type(GV) == View:
		view = GV
		graph = view.graph()
	else:
		graph = GV
		view = View(graph)
		
	# TODO(ben): Implement
	
	return view