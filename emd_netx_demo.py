#Test Code to know about Network X and matching only two graphs based on avg shortest path length

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyemd import emd
#creating new graph
#G= nx.Graph();

G=nx.Graph()
G.add_edge('a','b',weight=12)
G.add_edge('a','c',weight=1)
G.add_edge('c','d',weight=33)
G.add_edge('c','e',weight=44)
G.add_edge('c','f',weight=57)
G.add_edge('a','d',weight=69)
G.add_edge('b','c',weight=1)

#creating new graph
H=nx.Graph()

#H.add_edge('a','b',weight=24)
#H.add_edge('a','c',weight=2)
H.add_edge('c','d',weight=5000)
H.add_edge('c','e',weight=88)
H.add_edge('c','f',weight=114)
H.add_edge('a','d',weight=130)
H.add_edge('b','c',weight=2)
#H.add_edge('b', 'd',weight=200)
#PLotting graph
plt.figure(1)
nx.draw(G)
plt.figure(2)
nx.draw(H)

sum_array = []
count_array = []

#creating list of graphs
graphs = []
graphs.append(G)
graphs.append(H)

#for each graph finding avg shortest path
for graph in graphs:
	sum1 = 0.0
	count_nodes = 0.0
	values = []
	for node in graph:
		length = nx.single_source_dijkstra_path_length(graph,node)
		value = min(x for x in length.values() if x != 0)
		for i in range(len(length.values())):
			values.append(length.values()[i]/value)
		
		sum1 += sum(values)
		count_nodes += 1.0
	sum_array.append(sum1)
	count_array.append(count_nodes)

print "result:"


#storing avg shortest path for each graph
signature = []
for i in range(len(count_array)):
	signature.append([sum_array[i]/count_array[i]])

#print "signature is "
print signature
sig_np = []
#for i in signature:
#	sig_np.append(np.array[i])
#dtype=np.int
'''z = np.zeros((2,2))
z /= z.max()'''
z = []
z.append([2.0])

j = emd(np.array(signature[0]), np.array(signature[1]),np.array(z))
print j
#matching graph similarity
'''for i in range(len(count_array)):
	for j in range(i+1, len(count_array)):
		if (signature[i] > (0.80 * signature[j]) and signature[i] < (1.20 * signature[j])):
			print "graphs "+ str(i) + " and " + str(j) + " are similar!"
'''
#applying emd for finding similarity
#plt.show()

