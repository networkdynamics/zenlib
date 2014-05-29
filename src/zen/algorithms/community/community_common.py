# Returns a list of the keys which have the maximal value in a dictionary
def keys_of_max_value(dct):
	it = dct.iteritems()
	try:
		key, val = it.next()
	except:
		return None
	max_val = val
	candidates = [key]

	while True:
		try:
			key, val = it.next()
			if val == max_val:
				candidates.append(key)
			elif val > max_val:
				candidates = [key]
				max_val = val
		except StopIteration:
			break

	return candidates

# Modifies a community table so that the community indices range from 0 to 
# N - 1. Returns the number of communities
def normalize_communities(comm_table):
	old_to_new_comms = {}
	max_cidx = 0

	for i in range(len(comm_table)):
		cidx = comm_table[i]
		if cidx in old_to_new_comms:
			new_cidx = old_to_new_comms[cidx]
			comm_table[i] = new_cidx
		else:
			old_to_new_comms[cidx] = max_cidx
			comm_table[i] = max_cidx
			max_cidx += 1

	return max_cidx
