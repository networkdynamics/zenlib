
from cpython cimport bool

import numpy as np
cimport numpy as np

cpdef components(G)

cpdef np.ndarray[np.float_t, ndim=1] cddist(G,direction=*,bool inverse=*)

cpdef ddist(G,direction=*,bool normalize=*)

cpdef diameter(G)