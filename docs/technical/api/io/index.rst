Functions for loading and storing network data
==============================================

.. toctree::
	:hidden:
	
	bel.rst
	edgelist.rst
	gml.rst
	memlist.rst
	scn.rst

.. automodule:: zen.io

**Supported data formats**:

Zen supports the following standard network data formats:

	* :doc:`Edgelist <edgelist>`: this widely-used format encodes graph connectivity, but does not support node/edge attributes.
	* :doc:`Graph modeling language <gml>`: this format can encode graph connectivity as well as node/edge attributes of certain types.
	
Also supported are several more specialized formats that are particular to Zen alone:

	* :doc:`Binary edgelist <bel>`: this is a space efficient version of the edgelist that uses binary representation of 
	  nodes (rather than ascii representations).
	* :doc:`Memory-mapped edgelist <memlist>`: this storage format preserves the 
	  internal structure of a Zen graph (connectivity only).  The result is a format that can be read very quickly.  
	  This format is excellent for storing and loading very large networks.
	* The :doc:`Simple, Concise Network (SCN) format <scn>`: this storage format is an 
	  extension to the :doc:`edgelist <edgelist>` format in which an arbitrary number of attributes can also be specified 
	  for each node and edge.
	