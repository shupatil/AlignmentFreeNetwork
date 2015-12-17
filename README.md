# CSE 549 Comp Bio: Project: Alignment Free Network Comparison
# By Harsh Patel, Shubhada Patil, Anusha Ghanta, Deepika Periganji
# AlignmentFreeNetwork

This repo contains code for alignment free network comparison. There are two main files in this- eigen.py and line_graph.py

->How to run this code?

- Dataset can be generated using following command.
	python data_generator_rewire.py --N 100 --rho 0.1 --num 3 --seed 10

- then python line-test-graph1.py or python eigen.py

NOTE: for line graph signature, dataset is taken from the folder "datatset_3" only. ONe should generate dataset in that folder to run this code.

-> Time to run:

-Line graph signature is taking too much time because it is not coded in efficient manner. Signature is calculated every time so in order to avoid too much time, little modification of existing code is required. 
 
Precision recall curves can be found in this repo for different signatures. For line graph signature for 80 graphs(1000 nodes), running time was around one hour and for 280 graphs(100 nodes) it was around 25 minutes. To test it in less time one can try with 40 or 80 graphs (100 nodes each) which should take around 2 to 5 minutes. 
