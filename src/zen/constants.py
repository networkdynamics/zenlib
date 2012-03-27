"""
Constants that are used throughout Zen.
"""

# Graph directedness
DIRECTED = 'directed'
UNDIRECTED = 'undirected'

# Direction constants
BOTH_DIR = 'both_dir'
IN_DIR = 'in_dir'
OUT_DIR = 'out_dir'

# Constants for specifying how weights should be merged.
# These values are accepted by the DiGraph.skeleton function.
AVG_OF_WEIGHTS = 0
MAX_OF_WEIGHTS = 1
MIN_OF_WEIGHTS = 2

# Constants for specifying how data should be merged.
# These values are accepted by the DiGraph.skeleton function.
NO_NONE_LIST_OF_DATA = 0
LIST_OF_DATA = 1
