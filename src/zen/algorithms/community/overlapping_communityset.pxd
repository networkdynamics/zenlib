from zen.graph cimport Graph
from communityset cimport Community

from numpy cimport ndarray
from cpython cimport bool

cdef class OverlappingCommunitySet:
	cdef Graph _graph
	cdef object _communities
	cdef dict _probabilities
	cdef int _num_communities

	cdef void __raise_if_invalid_nidx(OverlappingCommunitySet self, int nidx) except *
	cdef __build_community(OverlappingCommunitySet self, int cidx)

	cpdef list node_communities_(OverlappingCommunitySet self, int nidx)
	cpdef list node_community_indices_(OverlappingCommunitySet self, int nidx)
	cpdef bool share_community_(OverlappingCommunitySet self, int u_idx, int v_idx)

