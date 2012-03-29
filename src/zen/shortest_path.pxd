from digraph cimport DiGraph
from graph cimport Graph

from cpython cimport bool
cimport numpy as np

cpdef dijkstra(G, start, end=?)
cpdef dijkstra_(G, int start_idx, int end_idx=?)
cpdef dijkstra_u_(Graph G, int start_idx, int end_idx=?)
cpdef dijkstra_d_(DiGraph G, int start_idx, int end_idx=?)

cpdef pred2path(start_obj, end_obj, R)
cpdef pred2path_(int start_idx, int end_idx, np.ndarray predecessor)

cpdef floyd_warshall(G)
cpdef floyd_warshall_(G)
cpdef floyd_warshall_u_(Graph G)
cpdef floyd_warshall_d_(DiGraph G)
