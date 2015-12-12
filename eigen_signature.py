import sys
def eigen_signature(vec):
	sum_vec = []
	vec_sorted = []
	'''no of eigen vectors provided'''
	len_vec = len(vec)
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
			#print "ratio of i: "+str(ratio)
		#print "k of i: "+str(k)
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
	vec = [[1,2,3,4,5,6,7,8,9],[4,3,6,5,15],[40,50,60,70,80,90]]
	print "given eigen vectors"
	print vec
	signatures = eigen_signature(vec)
	print "signatures are: "
	print signatures
if __name__ == "__main__":
	main()

