from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
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
	#sig1_sorted = (sorted(sig1,reverse=True))
	#sig2_sorted = (sorted(sig2,reverse=True))
	sig2_sorted = sig2
	sig1_sorted = sig1
	#print len(sig1)
	sum=0.0
	for i in range(0,k):
		diff = sig1_sorted[i] - sig2_sorted[i]
#		diff = np.subtract(np.array(sig1),np.array(sig2)))
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
count_filenames = 0
min_eval_len = 123456
model_names = []
#Reference Dataset
dic_list = []
for i in dataset:
	folder_name = '/home/deepika/AlignmentFreeNetwork/comsig-master/dataset_'+str(i) + '/'
	filenames = [f for f in listdir(folder_name)]
	for filename_out in filenames:
		count_filenames = count_filenames + 1
		#print "filename_out"+filename_out
		num_graphs=num_graphs+1;
		vec_name='vec'+str(i)
		evals_name='evals'+str(i)
		#model_name_o = filename_out.split('_')[0]
		model_names.append(filename_out.split('_')[0])
		Graph_List.append(folder_name+filename_out)
		graph = nx.read_edgelist(folder_name+filename_out)
		A = nx.adjacency_matrix(graph)
#		print (A.todense())
		evals_name,vec_name = np.linalg.eigh(A.todense())
		min_eval_len = min(min_eval_len,len(evals_name))
		vec=evals_name.tolist()
		evals_list.append(evals_name)
		sum_vec = 0.0
		vec_sorted = []
		'''no of eigen vectors provided'''
		len_vec = len(vec)
#		print "len_vec is" + str(len_vec)
#	    for i in range(0,len_vec):
#		vec_sorted.append(sorted(vec[i],reverse=True))
		for j in range(0,len(vec)):
			sum_vec = sum_vec+evals_name[j]
#		print "sum_vec"+ str(sum_vec) 
		''' Initializing the k_min - Analogous to top k '''
		k_min = sys.maxint
		k = 0
		sum_k = 0.0
		ratio = 0.0
		''' calculating k value for which energy is greater than 90% '''

#k=min_eval_len
#print model_names
k =10 
#print "size of evals_list"+str(len(evals_list))
#print 'graphs numebr '+str(num_graphs)
'''np.zeros because diagonal elements should be zero..rest are overwritten '''
distance_matrix = np.zeros((num_graphs,num_graphs))
#ref = np.ones(len(distance_matrix1))
distance_matrix1 = []
ref = []
count_compare = 0
for i in  range(0,num_graphs):
	for j in  range(i+1,num_graphs):
		count_compare = count_compare + 1
		ret=compare_sig(evals_list[i],evals_list[j],k)
#		print 'ret:'+str(ret)
#		print 'round:'+str(round(ret,8))
		distance_matrix[i][j] = ret
		distance_matrix[j][i] = distance_matrix[i][j]
		if (ret<10):
			distance_matrix1.append(0)
		else:
			distance_matrix1.append(1)
		if (model_names[i] == model_names[j]):
			ref.append(0)
		else:
			ref.append(1)
print "Ditance_matrix"
print distance_matrix1
#print "len of distancematix1 " + str(len(distance_matrix1))
#print "count_filenames "+ str(count_filenames)
#print "count_compare "+str(count_compare)
#ref = np.ones(len(distance_matrix1))

precision, recall, thresholds = precision_recall_curve(np.array(ref),distance_matrix1)
plt.plot(recall,precision,label='precision recall curve')
plt.show()
#average_precision = average_precision_score(np.array(ref),distance_matrix1)
#average_precision = f1_score(np.array(ref),distance_matrix1)
auc_score = roc_auc_score(np.array(ref),distance_matrix1)
print "Area under ROC curve:" + str(auc_score)
#print "average_precision "+str(average_precision)

if __name__ == "__main__":
        main()

#print evec
