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
