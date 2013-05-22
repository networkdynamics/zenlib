

    Explore
    Gist
    Blog
    Help

    jamie-public

    6
    6

public networkdynamics / zenlib

    Code
    Network
    Pull Requests 1
    Issues 3
    Wiki
    Graphs

    Tags

    Files
    Commits
    Branches 2

zenlib / src / setup.py
druths 3 months ago
Restored setup dependency to distutils and updated readme installatio…

1 contributor
file 80 lines (74 sloc) 4.535 kb
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 	

# try:
# from setuptools import setup
# except ImportError:
# #This should not be the case though
# from distutils.core import setup
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
Extension('zen.degree', ['zen/degree.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.hypergraph', ['zen/hypergraph.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.util.fiboheap', ['zen/util/fiboheap.pyx', 'zen/util/fib.c'], include_dirs=[numpy_include_dir]),
Extension('zen.util.queue', ['zen/util/queue.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.io.edgelist', ['zen/io/edgelist.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.io.hedgelist', ['zen/io/hedgelist.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.io.rdot', ['zen/io/rdot.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.io.scn', ['zen/io/scn.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.io.memlist', ['zen/io/memlist.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.algorithms.clustering', ['zen/algorithms/clustering.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.algorithms.components', ['zen/algorithms/components.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.randomize', ['zen/randomize.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.algorithms.centrality', ['zen/algorithms/centrality.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.layout.spring_layout', ['zen/layout/spring_layout.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.layout.random_layout', ['zen/layout/random_layout.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.layout.forceatlas_layout', ['zen/layout/forceatlas_layout.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.layout.fruchtermanreingold_layout', ['zen/layout/fruchtermanreingold_layout.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.algorithms.shortest_path', ['zen/algorithms/shortest_path.pyx'], include_dirs=[numpy_include_dir, fiboheap_include_dir]),
Extension('zen.algorithms.properties', ['zen/algorithms/properties.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.algorithms.modularity', ['zen/algorithms/modularity.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.algorithms.spanning', ['zen/algorithms/spanning.pyx'], include_dirs=[numpy_include_dir, fiboheap_include_dir]),
Extension('zen.algorithms.matching', ['zen/algorithms/matching.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.generating.rgm', ['zen/generating/rgm.pyx'], include_dirs=[numpy_include_dir]),
Extension('zen.algorithms.community.label_propagation', ['zen/algorithms/community/label_propagation.pyx'], include_dirs=[numpy_include_dir])]


setup(
name = 'Zen Library',
version = '0.9',
cmdclass = {'build_ext': build_ext},
ext_modules = ext_modules,
packages = ['zen','zen.data','zen.drawing','zen.generating','zen.io','zen.layout',
'zen.tests','zen.util','zen.algorithms.community','zen.benchmarks',
'zen.algorithms', 'zen.algorithms.flow'],
package_data = {'zen' : ['*.pxd'],
'zen.algorithms' : ['*.pxd'],
'zen.algorithms.community' : ['*.pxd'],
'zen.drawing' : ['*.pxd'],
'zen.data' : ['*.scn','*.elist','*.gml'],
'zen.generating' : ['*.pxd'],
'zen.io' : ['*.pxd'],
'zen.layout' : ['*.pxd'],
'zen.tests' : ['*.pxd','*.scn','*.elist','*.helist','*.rdot', '*.memlist', '*.gml'],
'zen.util' : ['*.pxd'],
'zen.data' : ['*.scn','*.gml'] },
# # dependencies
setup_requires = ['distribute','cython>=0.14'],
install_requires = ['numpy>=1.6.1','matplotlib>=1.0.1', 'networkx'],
#
# # testing suite
# test_suite = 'zen.test',
#
# # project metadata
author = 'Derek Ruths',
author_email = 'druths@networkdynamics.org',
description = 'Zen is a high-performance, easy-to-use network library developed specifically for Python.',
license = 'BSD',
url = 'http://zen.networkdynamics.org',
download_url = 'https://github.com/networkdynamics/zenlib'
)



GitHub
    About us
    Blog
    Contact & support
    GitHub Enterprise
    Site status

Applications
    GitHub for Mac
    GitHub for Windows
    GitHub for Eclipse
    GitHub mobile apps

Services
    Gauges: Web analytics
    Speaker Deck: Presentations
    Gist: Code snippets
    Job board

Documentation
    GitHub Help
    Developer API
    GitHub Flavored Markdown
    GitHub Pages

More
    Training
    Students & teachers
    The Shop
    Plans & pricing
    The Octodex

© 2013 GitHub, Inc. All rights reserved.

    Terms of Service
    Privacy
    Security


