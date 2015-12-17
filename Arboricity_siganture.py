from __future__ import division
import networkx as nx


g = nx.Graph()
g.add_edge(1,2);g.add_edge(1,3)
g.add_edge(1,7);g.add_edge(2,4)
g.add_edge(3,4);g.add_edge(3,5)
g.add_edge(3,6);g.add_edge(4,5)
g.add_edge(5,6);g.add_edge(6,7)

h=nx.Graph()
h.add_edge(1,2);h.add_edge(1,3)
h.add_edge(1,7);h.add_edge(2,4)
h.add_edge(3,4);h.add_edge(3,5)
h.add_edge(3,6);h.add_edge(4,5)
h.add_edge(5,6)

import itertools
n1=7
n2=6
graphList = [(g,n1),(h, n2)]
print len(graphList)
results2=[]
for graphindex in range(0,len(graphList)):
    print "in while"
    n=graphList[graphindex][1]
    max=0
    print "n is",n
    for j in range(2,n):
        target = nx.complete_graph(j)
        for sub_nodes in itertools.combinations(graphList[graphindex][0].nodes(),len(target.nodes())):
            subg = graphList[graphindex][0].subgraph(sub_nodes)
            if nx.is_connected(subg):
                print subg.edges()
                m=len(subg.edges())/(n-1)
                if max<m:
                    max=m
    #results1 = [graphList[graphindex][0].to_string(),max]
    results2.append(max)

print results2


