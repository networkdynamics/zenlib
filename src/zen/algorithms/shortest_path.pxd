from zen.digraph cimport DiGraph
from zen.graph cimport Graph

from cpython cimport bool
cimport numpy as np

cpdef single_source_shortest_path(G,source,target=?)
cpdef single_source_shortest_path_length(G,source,target=?)
cpdef single_source_shortest_path_(G,int source,int target=?)
cpdef single_source_shortest_path_length_(G,int source,int target=?)

cpdef dijkstra_path(G, start, end=?)
cpdef dijkstra_path_length(G, start, end=?)
cpdef dijkstra_path_(G, int start_idx, int end_idx=?)
cpdef dijkstra_path_length_(G, int start_idx, int end_idx=?)

cpdef bellman_ford_path(G, start)
cpdef bellman_ford_path_length(G, start)
cpdef bellman_ford_path_(G, int start_idx)
cpdef bellman_ford_path_length_(G, int start_idx)

cpdef pred2path(start_obj, end_obj, R)
cpdef pred2path_(int start_idx, int end_idx, np.ndarray predecessor)

cpdef floyd_warshall_path(G)
cpdef floyd_warshall_path_length(G)
cpdef floyd_warshall_path(G)
cpdef floyd_warshall_path_length_(G)

cpdef all_pairs_shortest_path(G)
cpdef all_pairs_shortest_path_length(G)
cpdef all_pairs_shortest_path_(G)
cpdef all_pairs_shortest_path_length_(G)

cpdef all_pairs_dijkstra_path(G)
cpdef all_pairs_dijkstra_path_length(G)
cpdef all_pairs_dijkstra_path_(G)
cpdef all_pairs_dijkstra_path_length_(G)

cpdef all_pairs_bellman_ford_path(G)
cpdef all_pairs_bellman_ford_path_length(G)
cpdef all_pairs_bellman_ford_path_(G)
cpdef all_pairs_bellman_ford_path_length_(G)


