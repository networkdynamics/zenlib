Zen Library Developer Documentation
===================================

*This section is still under development.*  Some place-holding text and tips have been added that need to be refined and 
organized in subsections and subpages.

Some quick notes on development:

	* When returning a group of objects, you have the option of using a :py:class:`list` or a :py:class:`set`.  If a major
	  way that the group of objects will be used is to test membership, then use a :py:class:`set`.  Otherwise, a
	  :py:class:`list` should be favored by default.
	
	* Documentation structure
	
		* *A function or method*.  A function's documentation has the following parts (in this sequence).
			* *Prototype*: when the function name and arguments are given in the prototype, it is best to
			  list all arguments (both positional and keywords).  Positional arguments should be given first.
			  Keyword arguments should be given second and should be enclosed in square braces. Any argument
			  that has a default value should have the default value shown in the prototype.  If there are too
			  many keyword arguments to give, drop the least-likely to be used and indicate the omission of them
			  with an ellipsis (``...``).
			* *Synopsis*: a brief, one sentence description of what the function does.
			* *Detailed description*: a more lengthy (1 or more paragraph) description of the behavior
			  of the function.  This may include citations to articles on which the function is based.
			* *Arguments*: arguments are of two kinds.  Positional arguments (*Args*) and keyword arguments
			  (*KwArgs*).  These are documented in two different lists (first positional, then keywords) which
			  are each preceded by the appropriate heading (either **Args** or **KwArgs**).  The list should
			  contain a list of the arguments.  Each entry in the list should have the following structure
			
				<argument_name> [=<default value>] (<argument_type>): <description>
				
			  Wherever possible, the ``argument_type`` should link to the documentation for that object type.
			* *Return value*: if the function returns a value, then this is documented under the heading **Returns**.
			  The structure of the description should be
			
				<return_type>, <reference_var_name>. <description>
				
			  The point of the ``reference_var_name`` is to have a placeholder for the return value that the description
			  string can use to explain the structure of the return value.  Wherever possible, the ``return_type`` should
			  link to the documentation for that object type.
			* *Exceptions raised*: if the function may raise an exception or error, then a list of the different exceptions is given.
			  If only one exception may be raised, then no list is necessary.  The entry for a single exception should have the
			  following structure
			
				<exception_type>: <description>
				
			  Wherever possible, the ``exception_type`` should link to the documentation for that exception.