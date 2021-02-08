import multiprocessing

INPUT_FILE = 'data/adj.tsv'
OUTPUT_FILE = 'res'
# TODO define P and Q
P=1.4 # P -> we keep the walk not too near to the starting node
Q=1.5 # Q higher than 1 ->  the random walk is biased towards nodes close to node t (considering that we are in a node v after t->v)
# TODO define WALK params
NUM_WALKS = 1
WALK_LENGTH = 30
# TODO define word2vec params
DIMENSIONS = 100
WINDOW_SIZE = 7
CORES = multiprocessing.cpu_count()
ITER = 1 							# number of epochs in SDG
# paths to save files
DATASET_FILE_NAME = 'data/DG-Miner_miner-disease-gene.tsv'
ADJ_FILE_NAME = 'data/adj.tsv'
TRAIN_EDGES_FILE_NAME = 'data/train_edges.tsv'
TEST_EDGES_FILE_NAME = 'data/test_edges.tsv'
VALIDATION_EDGES_FILE_NAME = 'data/validation_edges.tsv' 

# nn model 
BATCH_SIZE = 10000