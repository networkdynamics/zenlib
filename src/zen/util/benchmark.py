import time
import inspect
import sys
import warnings

class Benchmark:
	
	def __init__(self,name,number=3):
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
			print '\tTest: %s' % name[:10].ljust(12,' '),
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
			order = self.times.keys()
			order.sort(cmp=lambda x,y: -cmp(x,y))
			
		names = order

		data = [self.times[k] for k in order]
		
		if not raw:
			max_val = max(data)
			data = [(max_val/x) for x in data]
			
		
		width = 0.8
		pylab.bar(range(len(data)),data,width=0.8)
		pylab.title(self.name,fontsize=20,fontweight='bold')
		pylab.ylabel('Speed boost',fontsize=18,fontweight='bold')
		
		locs,labels = pylab.yticks()
		pylab.yticks(locs,['%sx' % str(int(x)) for x in locs])
		
		# TODO: Make bar labels
		pylab.xticks([x+width/2.0 for x in range(len(data))],order,fontsize=18,fontweight='bold')
			
def main():
	
	args = sys.argv
	
	gen_fig_files = False
	if len(args) == 2 and args[1] == 'plots':
		gen_fig_files = True
		import pylab as pl
		
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
			b.plot_bars()
			pl.savefig('%s.png' % b.name.replace(' ','_'))
			