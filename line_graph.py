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
    if G.is_directed():
        L = _lg_directed(G, create_using=create_using)
    else:
        L = _lg_undirected(G, selfloops=False, create_using=create_using)
    return L

def _node_func(G):
    if G.is_multigraph():
        def sorted_node(u, v, key):
            return (u, v, key) if u <= v else (v, u, key)
    else:
        def sorted_node(u, v):
            return (u, v) if u <= v else (v, u)
    return sorted_node

def _edge_func(G):
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
            L.add_node(nodes[0])

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
    for i in range(0,len(sig1)):
	sum1 += abs(sig1[i]-sig2[i])
    sum1 = sum1/(max1*len(sig1)) 
    return 1-sum1
    
def gen_sig(Graph):
    gl = line_graph(Graph);
    degree_sequence=nx.degree(gl) # degree sequence
    s = degree_sequence.values()
    s = sorted(s, reverse=True)
    return s

def main():
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
        for filename_inner in filenames:
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

    print "area under curve: ",roc_auc_score(np.array(ref),np.array(distance_matrix1))
    precision, recall, thresholds = precision_recall_curve(np.array(ref), np.array(distance_matrix1))
    plt.plot(recall,precision,label='precision recall curve')    
    plt.show()



if __name__ == "__main__":
    main()

