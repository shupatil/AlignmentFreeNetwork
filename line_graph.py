from __future__ import division
from optparse import OptionParser
from collections import deque
from math import log
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
import argparse
import os
from os import listdir
import sys,time,logging,random
import snap
import numpy as np
import matplotlib
from subprocess import call
import networkx as nx
import csv
import matplotlib.pyplot as plt
import pylab
from scipy.spatial.distance import *

def line_graph(G, create_using=None):
    """Returns the line graph of the graph or digraph ``G``.

    The line graph of a graph ``G`` has a node for each edge in ``G`` and an
    edge joining those nodes if the two edges in ``G`` share a common node. For
    directed graphs, nodes are adjacent exactly when the edges they represent
    form a directed path of length two.

    The nodes of the line graph are 2-tuples of nodes in the original graph (or
    3-tuples for multigraphs, with the key of the edge as the third element).

    For information about self-loops and more discussion, see the **Notes**
    section below.

    Parameters
    ----------
    G : graph
        A NetworkX Graph, DiGraph, MultiGraph, or MultiDigraph.

    Returns
    -------
    L : graph
        The line graph of G.
"""
    if G.is_directed():
        L = _lg_directed(G, create_using=create_using)
    else:
        L = _lg_undirected(G, selfloops=False, create_using=create_using)
    return L

def _node_func(G):
    """Returns a function which returns a sorted node for line graphs.

    When constructing a line graph for undirected graphs, we must normalize
    the ordering of nodes as they appear in the edge.

    """
    if G.is_multigraph():
        def sorted_node(u, v, key):
            return (u, v, key) if u <= v else (v, u, key)
    else:
        def sorted_node(u, v):
            return (u, v) if u <= v else (v, u)
    return sorted_node

def _edge_func(G):
    """Returns the edges from G, handling keys for multigraphs as necessary.

    """
    if G.is_multigraph():
        def get_edges(nbunch=None):
            return G.edges_iter(nbunch, keys=True)
    else:
        def get_edges(nbunch=None):
            return G.edges_iter(nbunch)
    return get_edges

def _sorted_edge(u, v):
    return (u, v) if u <= v else (v, u)

def _lg_directed(G, create_using=None):
    """Return the line graph L of the (multi)digraph G.

    Edges in G appear as nodes in L, represented as tuples of the form (u,v)
    or (u,v,key) if G is a multidigraph. A node in L corresponding to the edge
    (u,v) is connected to every node corresponding to an edge (v,w).

    Parameters
    ----------
    G : digraph
        A directed graph or directed multigraph.
    create_using : None
        A digraph instance used to populate the line graph.

    """
    if create_using is None:
        L = G.__class__()
    else:
        L = create_using

    # Create a graph specific edge function.
    get_edges = _edge_func(G)

    for from_node in get_edges():
        # from_node is: (u,v) or (u,v,key)
        L.add_node(from_node)
        for to_node in get_edges(from_node[1]):
            L.add_edge(from_node, to_node)

    return L

def _lg_undirected(G, selfloops=False, create_using=None):
    """Return the line graph L of the (multi)graph G.

    Edges in G appear as nodes in L, represented as sorted tuples of the form
    (u,v), or (u,v,key) if G is a multigraph. A node in L corresponding to
    the edge {u,v} is connected to every node corresponding to an edge that
    involves u or v.

    Parameters
    ----------
    G : graph
        An undirected graph or multigraph.
    selfloops : bool
        If `True`, then self-loops are included in the line graph. If `False`,
        they are excluded.
    create_using : None
        A graph instance used to populate the line graph.

    Notes
    -----
    The standard algorithm for line graphs of undirected graphs does not
    produce self-loops.

    """
    if create_using is None:
        L = G.__class__()
    else:
        L = create_using

    # Graph specific functions for edges and sorted nodes.
    get_edges = _edge_func(G)
    sorted_node = _node_func(G)

    # Determine if we include self-loops or not.
    shift = 0 if selfloops else 1

    edges = set([])
    for u in G:
        # Label nodes as a sorted tuple of nodes in original graph.
        nodes = [ sorted_node(*x) for x in get_edges(u) ]

        if len(nodes) == 1:
            # Then the edge will be an isolated node in L.
            L.add_node(nodes[0])

        # Add a clique of `nodes` to graph. To prevent double adding edges,
        # especially important for multigraphs, we store the edges in
        # canonical form in a set.
        for i, a in enumerate(nodes):
            edges.update([ _sorted_edge(a,b) for b in nodes[i+shift:] ])

    L.add_edges_from(edges)
    return L

def compare_sig(sig1, sig2):
    max1 = 0
    sum1 = 0
    if(len(sig1) > len(sig2)):
	for i in range(0,len(sig1)-len(sig2)):
	    sig2.append(0)
    else:
	for i in range(0,len(sig2)-len(sig1)):
	    sig1.append(0)
    max1 = max(sig1)
    if(max(sig2) > max1):
	max1 = max(sig2)
#    print "max1:", max1
    for i in range(0,len(sig1)):
	sum1 += abs(sig1[i]-sig2[i])
    sum1 = sum1/(max1*len(sig1)) 
#    print sig1, sig2, sum1
    return 1-sum1
    
def gen_sig(Graph):
    #print nx.is_connected(Graph)
    #gc = min(nx.connected_component_subgraphs(Graph))
    #plt.figure(2)
    #nx.draw(gc)
    #plt.show()

    #gl = line_graph(gc);
    gl = line_graph(Graph);
    #plt.figure(2)
    #nx.draw(gl)
    #plt.show()

    degree_sequence=nx.degree(gl) # degree sequence
    s = degree_sequence.values()
    s = sorted(s, reverse=True)
    #print "Degree sequence1", s
    return s

def main():


    # This is the first copy of the graph without any rewiring
    #print 'Drawing the Graph'
    #Gnet_X = nx.read_edgelist("./dataset_3/ER_1000_0.01_1.txt")
    #plt.figure(1)
    #nx.draw(Gnet_X)    
    #plt.show()
    #s = gen_sig(Gnet_X)
    #dataset = [1,3]
    
    #Gnet_Y = nx.read_edgelist("./dataset_3/ER_1000_0.01_2.txt")
    #plt.figure(1)
    #nx.draw(Gnet_X)    
    #plt.show()
    #t = gen_sig(Gnet_Y)
    distance_matrix = np.ones((40,40))
    distance_matrix1 = []
    Graph_List = []
    #for i in dataset:
    folder_name = './dataset_3' + '/'
    filenames = [f for f in listdir(folder_name)]
    i = 0
    j = 0
    ref = []
    for filename_out in filenames:
        model_name_o = filename_out.split('_')[0]
        Graph_List.append(folder_name+filename_out)
	G_1 = nx.read_edgelist(folder_name+filename_out)
        s = gen_sig(G_1)
	#print "filename 1:",filename_out
        for filename_inner in filenames:
	    #key_1 = filename_out + filename_inner
	    #key_2 = filename_inner + filename_out
	    #if not(key_1 in dic_list) and not(key_2 in dic_list):
		#    dic_list.append(key_1)
		 #   dic_list.append(key_2)
            model_name_i = filename_inner.split('_')[0]
            if model_name_i == model_name_o:
		ref.append(1)
		if not(filename_inner==filename_out):
		    G_2 = nx.read_edgelist(folder_name+filename_inner)
	            t = gen_sig(G_2)
	            sim = compare_sig(s,t)
     		    distance_matrix1.append(sim)
		else:
		    distance_matrix1.append(1)
	    else:
		ref.append(0)
		G_2 = nx.read_edgelist(folder_name+filename_inner)
	        t = gen_sig(G_2)
	        sim = compare_sig(s,t)
     		distance_matrix1.append(sim)

	    #if(j >= i):
                #print "filename 2:",filename_inner
	    #    G_2 = nx.read_edgelist(folder_name+filename_inner)
	    #    t = gen_sig(G_2)
	    #    sim = compare_sig(s,t)
                #print sim
	    #    distance_matrix[i][j] = sim
	#	if (sim < 0.95):
	#	    distance_matrix[i][j] = sim - 0.5
	#	distance_matrix[j][i] = distance_matrix[i][j]
	 #       distance_matrix1.append(distance_matrix[i][j])
	  #      if(model_name_i == model_name_o):
	   #         ref.append(1)
	    #    else:
		#    ref.append(0)
	    #j = j+1
        #j = 0
        #i = i+1

    print "ref is ",ref
    #print distance_matrix1
    #for i in range(0,20):
	#print distance_matrix[i]
    print "auc: ",roc_auc_score(np.array(ref),np.array(distance_matrix1))
    precision, recall, thresholds = precision_recall_curve(np.array(ref), np.array(distance_matrix1))
    plt.plot(recall,precision,label='precision recall curve')    
    plt.show()


"""
    distance_matrix = np.ones((2,2))
    for i in range (0,2):
	for j in range(i+1, 2):
	    sim = compare_sig(s,t)
            print sim
	    distance_matrix[i][j] = sim
	    distance_matrix[j][i] = distance_matrix[i][j]

            #if(sim<0.98):
	#	distance_matrix1.append(0)
	 #   else:
	#	distance_matrix1.append(1)

   # ref = np.ones(len(distance_matrix1))

    print distance_matrix
    #print "count is : ",count

    precision, recall, thresholds = precision_recall_curve(distance_matrix1, ref)
    plt.plot(recall,precision,label='precision recall curve')
    plt.show()
"""


if __name__ == "__main__":
    main()

