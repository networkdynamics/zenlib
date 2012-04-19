from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os.path
import sys
import numpy.core


numpy_include_dir = os.path.join(os.path.dirname(numpy.core.__file__),'include')
fiboheap_include_dir = os.path.join('zen','util')

ext_modules = [	Extension('zen.graph', ['zen/graph.pyx'], include_dirs=[numpy_include_dir]),
				Extension('zen.digraph', ['zen/digraph.pyx'], include_dirs=[numpy_include_dir]),
				Extension('zen.bipartite', ['zen/bipartite.pyx'], include_dirs=[numpy_include_dir]),
				Extension('zen.hypergraph', ['zen/hypergraph.pyx'], include_dirs=[numpy_include_dir]),
				Extension('zen.util.fiboheap', ['zen/util/fiboheap.pyx', 'zen/util/fib.c'], include_dirs=[numpy_include_dir]),
				Extension('zen.util.queue', ['zen/util/queue.pyx'], include_dirs=[numpy_include_dir]),
				Extension('zen.io.edgelist', ['zen/io/edgelist.pyx'], include_dirs=[numpy_include_dir]),
				Extension('zen.io.hedgelist', ['zen/io/hedgelist.pyx'], include_dirs=[numpy_include_dir]),
				Extension('zen.io.rdot', ['zen/io/rdot.pyx'], include_dirs=[numpy_include_dir]),
				Extension('zen.io.scn', ['zen/io/scn.pyx'], include_dirs=[numpy_include_dir]),
				Extension('zen.io.memlist', ['zen/io/memlist.pyx'], include_dirs=[numpy_include_dir]),
				Extension('zen.clustering', ['zen/clustering.pyx'], include_dirs=[numpy_include_dir]),
				Extension('zen.randomize', ['zen/randomize.pyx'], include_dirs=[numpy_include_dir]),
				Extension('zen.centrality', ['zen/centrality.pyx'], include_dirs=[numpy_include_dir]),
				Extension('zen.layout.spring_layout', ['zen/layout/spring_layout.pyx'], include_dirs=[numpy_include_dir]),
				Extension('zen.layout.random_layout', ['zen/layout/random_layout.pyx'], include_dirs=[numpy_include_dir]),
				Extension('zen.layout.forceatlas_layout', ['zen/layout/forceatlas_layout.pyx'], include_dirs=[numpy_include_dir]),				
				Extension('zen.layout.fruchtermanreingold_layout', ['zen/layout/fruchtermanreingold_layout.pyx'], include_dirs=[numpy_include_dir]),								
				Extension('zen.shortest_path', ['zen/shortest_path.pyx'], include_dirs=[numpy_include_dir, fiboheap_include_dir]),
				Extension('zen.properties', ['zen/properties.pyx'], include_dirs=[numpy_include_dir]),
                Extension('zen.modularity', ['zen/modularity.pyx'], include_dirs=[numpy_include_dir]),
				Extension('zen.spanning', ['zen/spanning.pyx'], include_dirs=[numpy_include_dir, fiboheap_include_dir]),
				Extension('zen.matching', ['zen/matching.pyx'], include_dirs=[numpy_include_dir]),
				Extension('zen.generating.simple', ['zen/generating/simple.pyx'], include_dirs=[numpy_include_dir]),
                Extension('zen.community.label_propagation', ['zen/community/label_propagation.pyx'], include_dirs=[numpy_include_dir])]


setup(
  name = 'Zen Network Library',
  url = 'http://zen.ruthsresearch.org',
  download_url = 'http://zenlib.googlecode.com/svn/trunk/',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
  packages = ['zen','zen.drawing','zen.generating','zen.io','zen.layout','zen.tests','zen.util','zen.community','zen.benchmarks','zen.data'],
  package_data = {	'zen' : ['*.pxd'],
					'zen.drawing' : ['*.pxd'],
					'zen.generating' : ['*.pxd'],
					'zen.io' : ['*.pxd'],
					'zen.layout' : ['*.pxd'],
					'zen.tests' : ['*.pxd','*.scn','*.elist','*.helist','*.rdot', '*.memlist'],
					'zen.util' : ['*.pxd'],
					'zen.data' : ['*.scn','*.gml'] }
  
)
