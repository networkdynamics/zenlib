
def type_check(obj,target_type,exception_msg=None):
	"""
	Check the type of a given object and raise a TypeError is the type isn't right.
	"""
	if type(obj) != target_type:
		if exception_msg == None:
			exception_msg = 'Received %s, type %s.  Expected %s' % (str(obj),str(type(obj)),str(target_type))
		raise TypeError, exception_msg
		
	return

class ZenException(Exception):
	pass
	
class InvalidGraphTypeException(ZenException):
	pass
	
class GraphChangedException(ZenException):
	pass