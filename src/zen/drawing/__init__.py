"""
The ``zen.drawing`` package contains modules that support drawing graph views to various drawing surfaces/devices.

Matplotlib figures
~~~~~~~~~~~~~~~~~~

.. automodule:: zen.drawing.mpl
"""

try:
	import pylab
	import mpl
except:
	print 'zen.drawing.mpl not auto-imported due to missing or broken matplotlib library'	
