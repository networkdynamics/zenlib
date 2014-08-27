import overlapping_communityset as ocs

import numpy as np

def freq_table_from_memory(memory):
	# Returns a frequency table (a list) of this memory
	# The ith element of the list is the frequency of the
	# ith element in the memory

	freqs = [freq for node, freq in memory]
	memory_sum = float(sum(freqs))
	for i in range(len(freqs)):
		freqs[i] /= memory_sum
	return freqs

def speak(memory):
	# Speaking rule for a node: select a random node
	# based on the probabilities given by the frequency table 
	table = freq_table_from_memory(memory)
	result = np.random.multinomial(1, table)

	return memory[np.argmax(result)][0]

def listen(nodes):
	# Listening rule for a node: select the most popular node
	# spoken by the neighbors

	current_max = nodes[0]
	node_counts = {nodes[0]: 1}
	for node in nodes[1:]:
		if node in node_counts:
			node_counts[node] += 1
		else:
			node_counts[node] = 1

		if node_counts[node] > node_counts[current_max]:
			current_max = node

	return current_max

def memorize(memory, new_node):
	# Add an occurence of new_node to the memory

	for i, (node, count) in enumerate(memory):
		if new_node == node:
			memory[i] = (memory[i][0], memory[i][1] + 1)
			return

	memory.append((new_node, 1))

def forget(memories, cutoff):
	# Remove from memory the node occurences whose frequency is lower than some
	# cutoff
	for i, memory in enumerate(memories):

		# Find which elements are below the cutoff
		freqs = freq_table_from_memory(memory)
		to_delete = []
		for j, freq in enumerate(freqs):
			if freq < cutoff:
				to_delete.append(j)

		if len(to_delete) == len(freqs):
		# If we're about to forget the entire memory for this node, keep the
		# maximal element instead (artificially, it'll have a count of 1)
			memories[i] = [(memory[np.argmax(freqs)], 1)]
		else:
			# Otherwise proceed normally (to_delete is reversed so that its
			# elements remain valid indices)
			for j in reversed(to_delete):
				del memory[j]

def build_community_set(G, memories):
	# Construct an overlapping community set from the memories of the nodes

	labels_to_cidx = {}
	communities = []
	probs = {}
	for node, memory in enumerate(memories):
		freqs = freq_table_from_memory(memory)
		for i, (label, frequency) in enumerate(memory):
			if label not in labels_to_cidx:
				labels_to_cidx[label] = len(communities)
				communities.append(set())

			cidx = labels_to_cidx[label]
			communities[cidx].add(node)
			probs[(node, cidx)] = freqs[i]

	return ocs.OverlappingCommunitySet(G, np.array(communities), 
										len(communities), probs)

def slpa(G, num_iterations=25, cutoff=0.1):
	# TODO: Fix up and comment

	rnge = G.max_node_idx + 1

	# A[i] is the memory of node i: a list of tuples (j, k) such that node j
	# has been "heard" k times.
	memories = [[(i, 1)] for i in range(rnge)]
	node_order = np.arange(rnge)

	for iter_count in range(num_iterations):
		# Look at nodes in a random order
		np.random.shuffle(node_order)

		for i in node_order:
			# Let the neighbors speak on which node they want to propagate,
			# then decide which one of those we keep and memorize
			candidates = [speak(memories[nbr]) for nbr in G.neighbors_(i)]
			winner = listen(candidates)
			memorize(memories[i], winner)

	# Remove node memories that are less frequent than the cutoff
	forget(memories, cutoff)
	return build_community_set(G, memories)
