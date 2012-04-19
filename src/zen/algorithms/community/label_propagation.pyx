import numpy as np
cimport numpy as np
from zen.graph cimport Graph
from libc.stdlib cimport rand
from zen.exceptions import ZenException



__all__ = ['label_propagation', 'label_propagation_']


cdef np.ndarray[np.int_t, ndim=1] label_nodes(Graph graph, np.ndarray[np.int_t, ndim=1] node_labels):
	#Faiyaz print"in label nodes function"
	# To hold max_node_index + 1 basically
	cdef int upper_limit = graph.max_node_index + 1
	# First get the random order in which it should visit node_labels
	cdef np.ndarray[np.int_t, ndim=1] order = np.arange(upper_limit)

	# Only stop recursing once this remains 0 at the end
	cdef int number_of_changes = 0

	# Declare neighbour-related arrays here
	#cdef np.ndarray[np.int_t, ndim=1] neighbours
	cdef np.ndarray[np.int_t, ndim=1] neighbour_label_counts
	cdef np.ndarray[np.int_t, ndim=1] neighbour_labels
	cdef int neighbour_label
	cdef int maxCount
	cdef int next_node
	cdef int numMax
	cdef int maxLabel
	cdef double coinflip
	cdef int firstTime=1
	cdef int majorityLabel=0
	cdef int countCurrentLabelNeighbor
	cdef int RAND_MAX= 0x7fffffff
	cdef int degree
	cdef int i,j,ei,eidx
	# Identify the labels of each of the neighbours
	neighbour_label_counts = np.zeros(upper_limit, np.int)
	neighbour_labels = np.zeros(upper_limit, np.int)
	
	while(firstTime or number_of_changes>0):
		if(majorityLabel==1):
			break
		print 'Number of changes' +str(number_of_changes)
		majorityLabel=1
		firstTime=0
		number_of_changes=0
		np.random.shuffle(order)
		
		
		# For each node
		for i in range(upper_limit):
			last_index_of_neighbour_label=0
			
			#Faiyaz printi
			#Faiyaz print"starting loop"
			next_node = order[i]
			
				
			maxCount=0
			maxLabel=-1
			numMax=1
			# Get the neighbours of the node
			#neighbours = graph.neighbors_(next_node)
			# For each neighbour
			#for j in range(len(neighbours)):
			
			degree = graph.node_info[next_node].degree
			for ei in range(degree):
				eidx = graph.node_info[next_node].elist[ei]
				j = graph.endpoint_(eidx,next_node)
				
				#Faiyaz printj
				neighbour_label= node_labels[j]
				if(neighbour_label_counts[neighbour_label]==0):
					neighbour_labels[last_index_of_neighbour_label]=neighbour_label
					last_index_of_neighbour_label+=1
				neighbour_label_counts[neighbour_label]+=1
				
				if(neighbour_label_counts[neighbour_label]>maxCount):
					maxCount=neighbour_label_counts[neighbour_label]
					maxLabel=neighbour_label
					numMax=1
				elif(neighbour_label_counts[neighbour_label]==maxCount):
					numMax+=1
					# Flip a coin and with probability 1/numMax, take it
					coinflip=rand()*1.0/RAND_MAX
					if(coinflip<1.0/numMax):
						maxCount=neighbour_label_counts[neighbour_label]
						maxLabel=neighbour_label
			# Now change the current label if necessary
			if(maxLabel != node_labels[next_node]):
				
				number_of_changes+=1
				node_labels[next_node]=maxLabel
			#Faiyaz print 'Node index '+str(next_node)
			#Faiyaz print 'Node object '+graph.node_object(next_node)
			#Faiyaz print 'Number of neighboring communities '+str(last_index_of_neighbour_label)
			#Faiyaz print 'Current Community '+ str(node_labels[next_node])
			#Faiyaz print 'Neighbour in current community '+str(neighbour_label_counts[node_labels[next_node]])
			#if(neighbour_label_counts[node_labels[next_node]]<(last_index_of_neighbour_label+1)/2):
				
			#	majorityLabel=0
			# ZEROING THE NEIGHBOR LABELS
			for j in range(last_index_of_neighbour_label):
				neighbour_label_counts[neighbour_labels[j]]=0
		
		# Now that everything has been done for this iteration, check if majorityLabel condition is satisfied
		
		#Faiyaz print 'Checking majorityLabvel'
		
		
		for i in range(upper_limit):
			maxCount=0
			maxLabel=-1
			last_index_of_neighbour_label=0			
			countCurrentLabelNeighbor=0
			next_node = order[i]
			
			degree = graph.node_info[next_node].degree
			for ei in range(degree):
				eidx = graph.node_info[next_node].elist[ei]
				j = graph.endpoint_(eidx,next_node)
				neighbour_label= node_labels[j]
				if(neighbour_label_counts[neighbour_label]==0):
					neighbour_labels[last_index_of_neighbour_label]=neighbour_label
					last_index_of_neighbour_label+=1
				neighbour_label_counts[neighbour_label]+=1
				if(neighbour_label==node_labels[next_node]):
					countCurrentLabelNeighbor+=1
				if(neighbour_label_counts[neighbour_label]>maxCount):
					maxCount=neighbour_label_counts[neighbour_label]
					maxLabel=neighbour_label
					
			#
			
			if(maxLabel != node_labels[next_node]):
				if(maxCount>countCurrentLabelNeighbor):
					majorityLabel=0
					for k in range(last_index_of_neighbour_label):
						neighbour_label_counts[neighbour_labels[k]]=0
					break
					
					
		#
			for j in range(last_index_of_neighbour_label):
				neighbour_label_counts[neighbour_labels[j]]=0
			
			
				
				
				
			
			
				
				
				
				
			
		
			

		
	return(node_labels)
					
				
				
#
cdef np.ndarray[np.int_t, ndim=1] label_nodes_weighted(Graph graph, np.ndarray[np.int_t, ndim=1] node_labels):
	#Faiyaz print"in label nodes function"
	# To hold max_node_index + 1 basically
	cdef int upper_limit = graph.max_node_index + 1
	# First get the random order in which it should visit node_labels
	cdef np.ndarray[np.int_t, ndim=1] order = np.arange(upper_limit)

	# Only stop recursing once this remains 0 at the end
	cdef int number_of_changes = 0

	# Declare neighbour-related arrays here
	#cdef np.ndarray[np.int_t, ndim=1] neighbours
	cdef np.ndarray[np.int_t, ndim=1] neighbour_label_counts
	cdef np.ndarray[np.float_t, ndim=1] neighbour_weight_counts
	
	cdef np.ndarray[np.int_t, ndim=1] neighbour_labels
	cdef int neighbour_label
	cdef float maxCount
	cdef int next_node
	cdef int numMax
	cdef int maxLabel
	cdef double coinflip
	cdef int firstTime=1
	cdef int majorityLabel=0
	cdef float countCurrentLabelNeighbor
	cdef int RAND_MAX= 0x7fffffff
	cdef int degree
	cdef int i,j,ei,eidx
	# Identify the labels of each of the neighbours
	neighbour_label_counts = np.zeros(upper_limit, np.int)
	neighbour_weight_counts = np.zeros(upper_limit, np.float)
	
	neighbour_labels = np.zeros(upper_limit, np.int)
	
	while(firstTime or number_of_changes>0):
		if(majorityLabel==1):
			break
		print 'Number of changes' +str(number_of_changes)
		majorityLabel=1
		firstTime=0
		number_of_changes=0
		np.random.shuffle(order)
		
		
		# For each node
		for i in range(upper_limit):
			last_index_of_neighbour_label=0
			
			#Faiyaz printi
			#Faiyaz print"starting loop"
			next_node = order[i]
			
				
			maxCount=0
			maxLabel=-1
			numMax=1
			# Get the neighbours of the node
			#neighbours = graph.neighbors_(next_node)
			# For each neighbour
			#for j in range(len(neighbours)):
			
			degree = graph.node_info[next_node].degree
			for ei in range(degree):
				eidx = graph.node_info[next_node].elist[ei]
				j = graph.endpoint_(eidx,next_node)
				
				#Faiyaz printj
				neighbour_label= node_labels[j]
				if(neighbour_label_counts[neighbour_label]==0):
					neighbour_labels[last_index_of_neighbour_label]=neighbour_label
					last_index_of_neighbour_label+=1
				neighbour_label_counts[neighbour_label]+=1
				neighbour_weight_counts[neighbour_label]+=graph.weight_(eidx)
				
				
				if(neighbour_weight_counts[neighbour_label]>maxCount):
					maxCount=neighbour_weight_counts[neighbour_label]
					maxLabel=neighbour_label
					numMax=1
				elif(neighbour_weight_counts[neighbour_label]==maxCount):
					numMax+=1
					# Flip a coin and with probability 1/numMax, take it
					coinflip=rand()*1.0/RAND_MAX
					if(coinflip<1.0/numMax):
						maxCount=neighbour_weight_counts[neighbour_label]
						maxLabel=neighbour_label
			# Now change the current label if necessary
			if(maxLabel != node_labels[next_node]):
				
				number_of_changes+=1
				node_labels[next_node]=maxLabel
			#Faiyaz print 'Node index '+str(next_node)
			#Faiyaz print 'Node object '+graph.node_object(next_node)
			#Faiyaz print 'Number of neighboring communities '+str(last_index_of_neighbour_label)
			#Faiyaz print 'Current Community '+ str(node_labels[next_node])
			#Faiyaz print 'Neighbour in current community '+str(neighbour_label_counts[node_labels[next_node]])
			#if(neighbour_label_counts[node_labels[next_node]]<(last_index_of_neighbour_label+1)/2):
				
			#	majorityLabel=0
			# ZEROING THE NEIGHBOR LABELS
			for j in range(last_index_of_neighbour_label):
				neighbour_label_counts[neighbour_labels[j]]=0
				neighbour_weight_counts[neighbour_labels[j]]=0
				
		
		# Now that everything has been done for this iteration, check if majorityLabel condition is satisfied
		
		#Faiyaz print 'Checking majorityLabvel'
		
		
		for i in range(upper_limit):
			maxCount=0
			maxLabel=-1
			last_index_of_neighbour_label=0			
			countCurrentLabelNeighbor=0.0
			next_node = order[i]
			
			degree = graph.node_info[next_node].degree
			for ei in range(degree):
				eidx = graph.node_info[next_node].elist[ei]
				j = graph.endpoint_(eidx,next_node)
				neighbour_label= node_labels[j]
				if(neighbour_label_counts[neighbour_label]==0):
					neighbour_labels[last_index_of_neighbour_label]=neighbour_label
					last_index_of_neighbour_label+=1
				neighbour_label_counts[neighbour_label]+=1
				neighbour_weight_counts[neighbour_label]+=graph.weight_(eidx)
				
				if(neighbour_label==node_labels[next_node]):
					countCurrentLabelNeighbor+=graph.weight_(eidx)
				if(neighbour_weight_counts[neighbour_label]>maxCount):
					maxCount=neighbour_weight_counts[neighbour_label]
					maxLabel=neighbour_label
					
			#
			
			if(maxLabel != node_labels[next_node]):
				if(maxCount>countCurrentLabelNeighbor):
					majorityLabel=0
					for k in range(last_index_of_neighbour_label):
						neighbour_label_counts[neighbour_labels[k]]=0
						neighbour_weight_counts[neighbour_labels[k]]=0
						
					break
					
					
		#
			for j in range(last_index_of_neighbour_label):
				neighbour_label_counts[neighbour_labels[j]]=0
				neighbour_weight_counts[neighbour_labels[j]]=0
				
			
			
				
				
				
			
			
				
				
				
				
			
		
			

		
	return(node_labels)			
			
			
			
	
        
       
    

# Calls label_propagation_. Pass it a graph. It returns a dictionary keyed by the node object and #returns the community label
cpdef label_propagation(graph,consider_weight=False):
	#Faiyaz print"in the regular function"
	# Get the list of node indices returned from the underscore function
	# Make it into a dictionary
	dictionary = {}
	cdef np.ndarray[np.int_t, ndim=1] labels = label_propagation_(graph,consider_weight)
	for i in range(len(labels)):
		# Key = node object, value = label
		node_object = graph.node_object(i)
		if node_object is not None:
			dictionary[node_object] = labels[i]
	return dictionary

cpdef np.ndarray[np.int_t, ndim=1] label_propagation_(graph,consider_weight=False):
	#Faiyaz print"in the underscore function"
	# First, make an array for the node indices
	# Get the length of the array - max_node_index + 1
	cdef int num_indices = graph.max_node_index + 1
	cdef np.ndarray[np.int_t, ndim=1] node_indices = np.zeros(num_indices, np.int)
	cdef int weight=consider_weight
	# Now go through the nodes in the graph
	for i in range(num_indices):
		if graph.node_object(i) is not None:
			#Faiyaz printi
			node_indices[i] = i
		else:
			node_indices[i] = -1
    

    # Now the array has been filled ... label the nodes
	if type(graph) == Graph:
		if(consider_weight==False):
			node_indices=label_nodes(<Graph> graph, node_indices)
		else:
			node_indices=label_nodes_weighted(<Graph> graph, node_indices)
			
	else:
		raise ZenException, 'Graph type %s is not currently supported' % str(type(graph))
	
	'''
    # Labeling is done on node_indices, normalise
    # Make another array holding relationships between 0s 1s etc
    cdef np.ndarray[np.int_t, ndim=1] real_labels = np.zeros(num_indices, np.int)

    
	# Make all the zeros into negative ones ...
    for i in range(num_indices):
        real_labels[i] = -1

    # Now do the relabelling
    last_real_label = 0
    for i in range(num_indices):
        current_label = node_indices[i]
        # Check if that label has a new label already
        if real_labels[current_label] < 0:
            # It doesn't ... set this to the last real label
            real_labels[current_label] = last_real_label
            node_indices[i] = last_real_label
        else:
            # Just give it the new label
            node_indices[i] = real_labels[current_label]
	'''
	return node_indices
