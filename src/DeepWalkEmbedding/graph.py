import numpy as np
import networkx as nx
import random

import concurrent.futures

import consts as constants
from RandomWalkThread import RandomWalkTread

class Graph():
	def __init__(self, nx_G, p, q):
		self.G = nx_G
		self.p = p
		self.q = q

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print('Walk iteration:')
		for walk_iter in range(num_walks):
			print('{}/{}'.format(walk_iter+1,num_walks))
			random.shuffle(nodes)

			node_chunks = chunks(nodes, constants.CORES)
			print('chunks num: {}'.format(len(node_chunks)))
			print('chunk len: {}'.format(len(node_chunks[0])))
			
			threads = []
			for i in range(constants.CORES):
				t = RandomWalkTread(node_chunks[i], G, self.p, self.q)
				t.start()
				threads.append(t)
			for i in range(constants.CORES):
				threads[i].join()
				walks = walks + threads[i].result
			
			
		return walks


def chunks(lst, n):
	ret = []
	chunk_size = int(len(lst)/n) + 1
	for i in range(n):
		if((i+1)*chunk_size > len(lst)):
			ret.append(lst[i*chunk_size:len(lst)])
			break 
		ret.append(lst[i*chunk_size:(i+1)*chunk_size])
	return ret

