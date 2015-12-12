import os
import sys,time,logging,random
import networkx as nx
import numpy as np
from time import clock
from scipy.linalg import eigh as largest_eigh
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh

import matplotlib.pyplot as plt
import pylab
from scipy.spatial.distance import *
from pyemd import emd
from os import listdir
#Reference Dataset
def eigen_signature(vec):
        sum_vec = []
        vec_sorted = []
        '''no of eigen vectors provided'''
        len_vec = len(vec)
	print "len_vec is" + str(len_vec)
	#print vec
        for i in range(0,len_vec):
                sum_vec.append(0.0)

        for i in range(0,len_vec):
                '''Sorting the eigen vectors '''
                vec_sorted.append(sorted(vec[i],reverse=True))
                '''Calculating sum in each eigen vector '''
                for j in range(0,len(vec[i])):
                        sum_vec[i] = sum_vec[i] + vec[i][j]

        ''' Initializing the k_min - Analogous to top k '''
        k_min = sys.maxint
        for i in range(0,len_vec):
                k = 0
                sum_k = 0.0
                ratio = 0.0
                ''' calculating k value for which energy is greater than 90% '''
                while (ratio < 0.9):
                        sum_k = sum_k + vec_sorted[i][k]
                        ratio = sum_k/sum_vec[i]
                        k = k + 1
                        print "ratio of i: "+str(ratio)
                print "k of i: "+str(k)
                '''Finding min of all such k values '''
                k_min = min(k_min, k)
        print "k_min is: "+str(k_min)

        '''Getting the signatures which contain top k eigen values '''
        sig = []
        for i in range(0,len_vec):
                sig.append([])
                for j in range(0, k_min):
                        sig[i].append(vec_sorted[i][j])
	return sig

def main():
        '''Eigen vectors generated for 3 graphs '''
#        vec = [[1,2,3,4,5,6,7,8,9],[4,3,6,5,15],[40,50,60,70,80,90]]
#        print "given eigen vectors"
#        print vec
Graph_List = []
dataset = [3]
index = 0
#Reference Dataset
dic_list = []
ground_data = []
test_data = []
for i in dataset:
        folder_name = '/home/anusha/CompBio/AlignmentFreeNetwork/comsig-master/dataset_'+str(i) + '/'
        print folder_name
        filenames = [f for f in listdir(folder_name)]
        for filename_out in filenames:
            #print filename_out
            model_name_o = filename_out.split('_')[0]
            Graph_List.append(folder_name+filename_out)
#           print Graph_List
            graph = nx.read_edgelist(folder_name+filename_out)
L = nx.adjacency_matrix(graph)
evals,vec = np.linalg.eigh(L)
#ind = np.argsort(evals)
#evals = evals[ind]
#vec = vec[:,ind]
print "evals are "
#print vec
signatures = eigen_signature(vec)
print "signatures are: "
print signatures
if __name__ == "__main__":
        main()

#print evec
