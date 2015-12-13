import os
import sys,time,logging,random
import networkx as nx
import numpy as np
from time import clock
from scipy.linalg import eigh as largest_eigh
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from math import log
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import pylab
from scipy.spatial.distance import *
from pyemd import emd
from os import listdir

def compare_sig(sig1,sig2,k):
	sum=0.0
	for i in range(0,k):
		diff=sig1[i]-sig2[i]
		sum=sum+(diff*diff)
	return sum


#Reference Dataset
def main():
        '''Eigen vectors generated for 3 graphs '''
#        vec = [[1,2,3,4,5,6,7,8,9],[4,3,6,5,15],[40,50,60,70,80,90]]
#        print "given eigen vectors"
#        print vec
Graph_List = []
eigenk=[]
dataset = [3]
evals_list=[]
index = 0
num_graphs=0
Nodes=100
#Reference Dataset
dic_list = []
for i in dataset:
        folder_name = '/home/anusha/CompBio/AlignmentFreeNetwork/comsig-master/dataset_'+str(i) + '/'
        filenames = [f for f in listdir(folder_name)]
        for filename_out in filenames:
            print filename_out
	    num_graphs=num_graphs+1;
	    vec_name='vec'+str(i)
	    evals_name='evals'+str(i)
            model_name_o = filename_out.split('_')[0]
            Graph_List.append(folder_name+filename_out)
#           print Graph_List
            graph = nx.read_edgelist(folder_name+filename_out)
#	    nx.draw(graph)
#	    plt.show()
	    A = nx.adjacency_matrix(graph)
#	    print (A.todense())
	    '''
	    D = np.zeros((100, 100))
	    for i in range(0,100):
	    	deg = 0
        	for j in range(0,100):
            		deg = deg +  int(A[i,j])
        		D[i,i] = deg
	    L = np.zeros((100,100))

	    for i in range(0,100):
        	for j in range(0,100):
            		L[i,j] = D[i,j] - A[i,j]
	    '''
	    evals_name,vec_name = np.linalg.eigh(A)
#	    print evals_name
	    vec=evals_name.tolist()
	    evals_list.append(evals_name)
            sum_vec = 0.0
            vec_sorted = []
	    '''no of eigen vectors provided'''
            len_vec = len(vec)
            print "len_vec is" + str(len_vec)
#	    for i in range(0,len_vec):
#		vec_sorted.append(sorted(vec[i],reverse=True))
            for j in range(0,len(vec)):
                sum_vec = sum_vec+evals_name[j]
	    print "sum_vec"+ str(sum_vec) 
	    ''' Initializing the k_min - Analogous to top k '''
	    k_min = sys.maxint
	    k = 0
            sum_k = 0.0
            ratio = 0.0
            ''' calculating k value for which energy is greater than 90% '''
#            for p in range(0,len(vec)):
#		while (ratio < 0.9):
			#print "type of "
#              		print type(vec_sorted[i][k])
#			sum_k = sum_k + evals_name[p]
#			print "evals_name[p]"+str(evals_name[p])
#			print "sum_k" +str(sum_k)
#                	ratio = sum_k/sum_vec
#			print "ratio:"+ str(ratio)
#                        k = k + 1
#			print " k for p "+str(p)+ ":"+str(k) 
	    eigenk.append(k)

k=1
print 'graphs numebr '+str(num_graphs)
distance_matrix = np.ones((num_graphs,num_graphs))
distance_matrix1 = []
for i in  range(0,num_graphs):
	for j in  range(i+1,num_graphs):
		ret=compare_sig(evals_list[i],evals_list[j],k)
		print 'ret:'+str(ret)
		print 'round:'+str(round(ret,8))
		distance_matrix[i][j] = ret
            	distance_matrix[j][i] = distance_matrix[i][j]
		if (ret<1):
			distance_matrix1.append(0)
		else:
			distance_matrix1.append(1)
print distance_matrix1
ref = np.ones(len(distance_matrix1))
precision, recall, thresholds = precision_recall_curve(ref,distance_matrix1)
plt.plot(recall,precision,label='precision recall curve')
plt.show()
if __name__ == "__main__":
        main()

#print evec
