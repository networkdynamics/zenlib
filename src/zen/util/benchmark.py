"""
A core objective of Zen is to provide among the fastest network-based functionality available.  In order to do this, the Zen
library includes a benchmarking class.  Much like a unit test, a benchmark subclass defines a set of methods that run code
whose performance will be quantified.  Benchmarking is implicitly comparative - so the benchmark class is designed to make
comparison between the performance of the different benchmark methods easy.  Textual and graphical output can be produced
showing how the performance of the different benchmark functions performed.

The remainder of this document explains how the ``zen.util.benchmark.Benchmark`` class should be subclassed and used.

Initialization
--------------

A subclass should call the :py:meth:`Benchmark.__init__(name,number)` function.

.. automethod:: Benchmark.__init__(name,number=3)

Setup and teardown
------------------

Prior to running the individual benchmarks, the ``Benchmark.setup()`` method is called.  Any work done in this
method is not included in the timed computations.

After the benchmarks have been run, the ``Benchmark.teardown()`` method is called.  This function can do any cleanup
or releasing of resources that were allocated for the benchmarking functions.

Command-line Interface
----------------------

The ``zen.util.benchmark`` module provides a ``main()`` function that, when executed, finds 
all ``Benchmark`` subclasses within the current namespace, and runs the the benchmarks according
the the command-line arguments provided (note that the function accepts no arguments - all configuration
is assumed to come through ``sys.argv``).

Detailed instruction for the command-line interface can be found by providing the ``-h`` flag.  Of note,
the interface supports generating two kinds of output:

	* *Textual.* By default and in all cases, the tool will generate text output showing the results of
	  each benchmark run.
	
	* *Graphical.* When the ``plots`` or ``xplots`` commands are given, a histogram is generated for each
	  benchmark showing the relative performance of each benchmark method in a given benchmark.  The histogram
	  is saved in PNG format to the current working directory.
	
		* ``plots`` generates histograms in which the y-axis is the time taken by each benchmark method.
		
		* ``xplots`` generates histograms in which the y-axis is the order of improvement of each benchmark method
		  relative to the slowest benchmark.
	
"""

import time
import inspect
import sys
import warnings

class Benchmark:
	
	def __init__(self,name,number=3):
		"""
		Initialize the 
		"""
		self.number = number
		self.name = name
		self.tests = {}
		self.times = {}
		
		self.__load_benchmarks()
		
	def setup(self):
		pass
		
	def teardown(self):
		pass
		
	def __load_benchmarks(self):
		methods = dict(inspect.getmembers(self,inspect.ismethod))
		bm_names = map(lambda x: x[3:], filter(lambda x: x.startswith('bm_'),methods.keys()))
		self.max_name_len = max([len(x) for x in bm_names])
		
		for n in bm_names:
			test_fxn = methods['bm_%s' % n]
			setup_name = 'setup_%s' % n
			if setup_name in methods:
				setup_fxn = methods[setup_name]
			else:
				setup_fxn = None
				
			teardown_name = 'teardown_%s' % n
			if teardown_name in methods:
				teardown_fxn = methods[teardown_name]
			else:
				teardown_fxn = None
				
			self.tests[n] = (setup_fxn,test_fxn,teardown_fxn)
			#print n, self.tests[n]
		
	def run(self):
		
		self.setup()
		
		print '\n%s:' % self.name
		for name,fxns in self.tests.items():
			run_times = []
			print '\tTest: %s' % name.ljust(self.max_name_len+2,' '),
			setup_fxn,test_fxn,teardown_fxn = fxns
			
			for r in range(self.number):
				if setup_fxn is not None:
					setup_fxn()
				
				time1 = time.time()
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					test_fxn()
				
				time2 = time.time()
				run_times.append(time2-time1)
				
				if teardown_fxn is not None:
					teardown_fxn()
				
			self.times[name] = min(run_times)
			print self.times[name]
			
		self.teardown()
			
	def plot_bars(self,order=None,raw=False):
		import pylab

		if order is None:
			# by default order from fastest to slowest
			order = self.times.items()
			order.sort(cmp=lambda x,y: cmp(x[1],y[1]))
			order = map(lambda x: x[0], order)
			
		names = map(lambda x: x.replace('_',' '),order)

		data = [self.times[k] for k in order]
		
		if not raw:
			max_val = max(data)
			data = [(max_val/x) for x in data]
			
		
		width = 0.8
		pylab.bar(range(len(data)),data,width=0.8)
		pylab.title(self.name,fontsize=20,fontweight='bold')
		
		if raw:
			pylab.ylabel('Elapsed time (sec)',fontsize=18,fontweight='bold')
		else:
			pylab.ylabel('Speed boost',fontsize=18,fontweight='bold')
		
			locs,labels = pylab.yticks()
			pylab.yticks(locs,['%sx' % str(int(x)) for x in locs])
		
		# TODO: Make bar labels
		pylab.xticks([x+width/2.0 for x in range(len(data))],names,fontsize=18,fontweight='bold')
			
def main():
	
	args = sys.argv
	
	gen_fig_files = False
	if len(args) == 2:
		if args[1] == 'xplots':
			gen_fig_files = True
			use_raw = False
			import pylab as pl
		elif args[1] == 'plots':
			gen_fig_files = True
			import pylab as pl
			use_raw = True
		
	caller = inspect.currentframe().f_back
	m_name = caller.f_globals['__name__']

	# get all the benchmarks
	benchmarks = []
	for o in inspect.getmembers(sys.modules[m_name], inspect.isclass):
		if inspect.isclass(o[1]) and o[1] != Benchmark:
			#print o,type(o),Benchmark,issubclass(o[1],Benchmark)
			benchmarks.append(o[1])
	
	# run each benchmark
	for i,benchmark in enumerate(benchmarks):
		b = benchmark()
		b.run()
		
		if gen_fig_files is True:
			pl.figure()
			b.plot_bars(raw=use_raw)
			pl.savefig('%s.png' % b.name.replace(' ','_'))
			