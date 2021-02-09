from threading import Thread
import consts as constants
import numpy as np

# the part of code relative to n2v that I will not use has been taken from 
# https://github.com/aditya-grover/node2vec

class RandomWalkTread(Thread):
    def __init__(self, nodes, G, p, q):
        Thread.__init__(self)
        self.result = []
        self.nodes = nodes
        self.G = G
        self.p = p
        self.q = q
    
    # n2v=True means that I have to use node2vec
    # n2v=Flase instead means that I have to use DeepWalk
    def node2vec_walk(self, start_node, n2v=False):
        
        G = self.G
        walk = [start_node]

        while len(walk) < constants.WALK_LENGTH:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if(n2v):
                    if len(walk) == 1:
                        
                        walk.append(cur_nbrs[int(np.floor(np.random.rand()*len(cur_nbrs)))])
                    else:
                        prev = walk[-2]
                        alias_edge = self.get_alias_edge(prev, cur)
                        next = cur_nbrs[alias_draw(alias_edge[0], alias_edge[1])]
                        walk.append(next)
                else:
                    walk.append(cur_nbrs[int(np.floor(np.random.rand()*len(cur_nbrs)))])

            else:
                break
        
        return walk

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(1/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(1)
            else:
                unnormalized_probs.append(1/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def run(self):
        for node in self.nodes:
            self.result.append(self.node2vec_walk(node))

    

def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]