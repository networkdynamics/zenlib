import numpy as np
from zen.graph import Graph
import communityset as cs
from community_common import keys_of_max_value

def propagate(G, label_ptable):
	new_label_ptable = [{} for i in range(G.max_node_idx + 1)]	
	for node in G.nodes_():
		num_neighbors = len(G.neighbors_(node))
		# Compute the sum of all probs, for all neighboring labels
		for nghbr in G.neighbors_(node):
			for label, prob in label_ptable[nghbr].iteritems():
				if label not in new_label_ptable[node]:
					new_label_ptable[node][label] = prob
				else:
					new_label_ptable[node][label] += prob

	return new_label_ptable

def normalize(G, label_ptable):
	# Normalize each label table so that they remain probability distributions
	for node in G.nodes_():
		sum_values = 0		
		for prob in label_ptable[node].itervalues():
			sum_values += prob

		for label, prob in label_ptable[node].iteritems():
			label_ptable[node][label] = prob / sum_values

def inflate(G, label_ptable, inflation):
	# Raise every probability to a power. This increases the gaps between
	# small and large probabilities.	
	for node in G.nodes_():

		to_remove = []

		for label, prob in label_ptable[node].iteritems():
			new_prob = prob ** inflation
			if new_prob < 0.00000001:
				new_prob = 0.0
				to_remove.append(label)
			label_ptable[node][label] = new_prob

		# Outright remove
		for label in to_remove:
			del label_ptable[node][label]

def cutoff(G, label_ptable, cutoff):
	# In all label tables, remove labels probabilities below a certain threshold
	for node in G.nodes_():
		to_delete = []
		for label, prob in label_ptable[node].iteritems():
			if prob < cutoff:
				to_delete.append(label)

		for label in to_delete:
			del label_ptable[node][label]
	#TODO Nodes might become unlabeled in this process. This needs to be fixed.

# Returns 1 if a is a subset of b, 0 otherwise
def is_subset(a, b):
	if len(a) > len(b):
		return 0

	i = 0
	for val in a:
		if val not in b:
			return 0

	return 1

# Create a community list from the label table, throwing away the probabilities
# and keeping the label with the highest probability
def create_community_list(G, label_ptable):
	communities = np.empty(len(label_ptable), np.int)
	for node in G.nodes_():
		# If we somehow have a tie, always pick the first community
		communities[node] = keys_of_max_value(label_ptable[node])[0]

	return communities

def label_rank(G, inflation=4.0, cutoff_thresh=0.1, cond_update=0.7, **kwargs):
	"""
	Detect communities in a graph using the LabelRank algorithm described in
	[Xie, Szymanski 2013]. It assigns a table to each node, containing the
	probabilities that this node has a certain label. At each step, this table
	is propagated to the node's neighbors. Then, it is inflated (raised to a
	certain power, which increases the differences between high and low
	probabilities). Entries below a given threshold are removed from the table.
	Finally, a node's table is only updated if it disagrees enough with its
	neighbors. The algorithm stops either when there have been no changes or
	when the same number of changes happened too many times.

	**Args**

		* ``inflation [=4.0]`` (float): the power to which the probabilities
			are to be raised after every propagation step. Higher values
			cause a larger difference in small and large probabilities.

		* ``cutoff_thresh [=0.1]`` (float): the cut-off below which inflated
			probabilities are removed from the table. Higher values limit
			propagation of low-probability labels.

		* ``cond_update [=0.7]`` (float): the percentage of agreement required
			between a node and its neighbors to not update this node anymore.

	**Keyword Args**

		* ``max_num_changes [=5]`` (int): the number of times a particular
			number of updates is allowed before stopping the algorithm. For
			example, with ``max_num_changes = 5``, the algorithm will stop
			if it detects that some number of updates occured 5 times.

        ## TODO: Switch -1 to None/Infinity
		* ``max_iterations [=-1]`` (int): if lgreater than or equal to zero, the
			algorithm will run at most this many iterations. Negative values
			indicate that the algorithm will run until normal completion.

    ## TODO: Add links to CommunitySet, etc..
	**Returns**
		A :py:module:?CommunitySet containing the communities detected in the graph. This
		is done by taking the label with maximal probability and assigning it to
		the node (in case of a tie, the first maximal label in the table is 
		used). Then, the communities are formed by groups of nodes with the same
		label.		
        
    ..seealso::
        Cite the paper here.
        
	"""
	max_iterations = kwargs.pop('max_iterations', -1)
	max_num_changes = kwargs.pop('max_num_changes', 5)
	
	label_ptable = [{} for i in range(G.max_node_idx + 1)]
	added_selfloop = []

	for node in G.nodes_():
		# The LabelRank paper suggests adding a self-loop to every node to
		# improve detection quality
		if not G.has_edge_(node, node):
			added_selfloop.append(node)
			G.add_edge_(node, node)

		initial_prob = 1.0 / len(G.neighbors_(node))
		for nghbr in G.neighbors_(node):
			label_ptable[node][nghbr] = initial_prob

	num_changes_table = {}
	i = 0
	while True:
		if max_iterations >= 0 and i == max_iterations:
			break

		new_label_ptable = propagate(G, label_ptable)
		inflate(G, new_label_ptable, inflation)
		normalize(G, new_label_ptable)
		cutoff(G, new_label_ptable, cutoff_thresh)
		normalize(G, new_label_ptable)

		# Conditional update over every node
		toUpdate = []
		num_changes = 0
		for node in G.nodes_():
			sum_subsets = 0
			ci = keys_of_max_value(label_ptable[node])
			for nghbr in G.neighbors_(node):
				# Do not consider ourselves when checking neighbors
				if nghbr == node: 
					continue			

				cj = keys_of_max_value(label_ptable[nghbr])
				sum_subsets += is_subset(ci, cj)

			if sum_subsets < (cond_update * len(G.neighbors_(node))):
				num_changes += 1
				toUpdate.append(node)

		for node in toUpdate:
			label_ptable[node] = new_label_ptable[node]

		normalize(G, label_ptable)

		i += 1
		# Stop criteria: either no change or the same number of changes happened
		# too many times
		if num_changes == 0:
			break

		if num_changes not in num_changes_table:
			num_changes_table[num_changes] = 1
		else:
			num_changes_table[num_changes] += 1

		if num_changes_table[num_changes] > max_num_changes:
			break

	# Remove the self-loops we added
	for node in added_selfloop:
		G.rm_edge_(G.edge_idx_(node, node))
	
	return cs.CommunitySet(G, create_community_list(G, label_ptable))
