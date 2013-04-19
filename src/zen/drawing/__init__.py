"""
The ``zen.drawing`` package contains modules that support drawing graph views to various drawing surfaces/devices.

Matplotlib figures
~~~~~~~~~~~~~~~~~~

.. automodule:: zen.drawing.mpl

Rendering to Ubigraph
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: zen.drawing.ubigraph
"""

try:
	import pylab
	import mpl
except:
	print 'zen.drawing.mpl not auto-imported due to missing or broken matplotlib library'
	
from ubigraph import UbigraphRenderer
