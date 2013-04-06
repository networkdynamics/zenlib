# This is a graph object in gml file format
# produced by the zen graph library

graph [
	directed 0
	bipartite 0
	node [
		id 0
		name "A"
		zenData [
			label "soft"
			attr1 "one"
			attr2 "two"
		]
	]
	node [
		id 1
		name "B"
		label "soft"
		attr1 "one"
		attr2 "two"
	]
	edge [
		id 0
		source 0
		target 1
		weight 1.0
		zenData [
			label "soft"
			attr1 "one"
			attr2 "two"
		]
	]
	edge [
		id 1
		source 1
		target 2
		weight 1.0
		zenData "included if other data present"
		label "soft"
		attr1 "one"
		attr2 "two"
	]
]
