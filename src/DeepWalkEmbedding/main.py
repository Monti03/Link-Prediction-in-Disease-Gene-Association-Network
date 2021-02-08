'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

# https://github.com/aditya-grover/node2vec
# Of this implementation I took the node2vec part that I will not use since it 
# will took to much time or memory.
# The deep walk implementation comes from an adaptation of the n2v code 

import numpy as np
import pandas as pd
import networkx as nx
from gensim.models import Word2Vec

import os

import graph
import consts as  constants

from utils import train_test_split, create_adj_matrix, split_dataset

def read_graph(input_file):
	'''
	Reads the input network in networkx.
	'''
	G = nx.read_edgelist(input_file, nodetype=int, create_using=nx.DiGraph())
	
	G = G.to_undirected()

	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [list(map(str, walk)) for walk in walks]
	print(len(walks))
	model = Word2Vec(walks, size=constants.DIMENSIONS, window=constants.WINDOW_SIZE, min_count=0, sg=1, workers=constants.CORES, iter=constants.ITER)
	model.wv.save_word2vec_format(constants.OUTPUT_FILE)
	
	model.save('word2vec.wordvectors')

	return

def main():
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''

	if (constants.INPUT_FILE == ''):
		print('no input file name')
		return 
	print('building graph')
	nx_G = read_graph(constants.INPUT_FILE)
	print('builded graph')
	G = graph.Graph(nx_G, constants.P, constants.Q)
	
	print('simulate walks')
	walks = G.simulate_walks(constants.NUM_WALKS, constants.WALK_LENGTH)
	print(len(walks))
	print('learn embeddings')
	learn_embeddings(walks)


if __name__ == "__main__":
	
	if not os.path.isfile('data/adj.tsv'):
		if not os.path.isfile('data/train_edges.tsv'):
			split_dataset()
		
	main()
