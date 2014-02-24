graph [
	comment "This is a sample graph" 
	label "Hello, I am a graph"
	
	node [
		id 1
		label "N1" 
		listVar 1
		listVar "a"
		listVar 3.2
	]
	node [
		id 2
		label "N2" 
		sampleTwo 42.1
	]
	node [
		id 3
		label "N3" 
		sampleThree "HELLO WORLD"
	]
	
	edge [
		source 1
		target 2
		label "Edge from node 1 to node 2"
		value 2
	]
	edge [
		source 2
		target 3
		label "Edge from node 2 to node 3"
		value 3
	]
]
