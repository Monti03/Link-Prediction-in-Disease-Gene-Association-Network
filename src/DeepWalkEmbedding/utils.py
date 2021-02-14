from scipy.sparse import csr_matrix
import scipy.sparse as sp
import pandas as pd
import numpy as np

import random

import consts as constants

def save_walks(walks):
    with open(constants.WALKS_FILE, 'w') as fout:
        for walk in walks:
            for node in walk:
                fout.write(str(node) + ' ')
            fout.write('\n')

def train_test_split(adj, num_genes, train_ratio=0.01, test_ratio=0.01):

    num_nodes = adj.shape[0] 
    
    # get upper triangular matrix (without the main diagonal, so k=1, otherwise 0)
    adj_triu = sp.triu(adj, k=1)
    
    # return row and col indices of non-zero values in net_triu
    row, col, _ = sp.find(adj_triu)
    
    # randomly permute edges
    perm = np.random.permutation(range(len(row)))
    row, col = row[perm], col[perm]
    
    # sample train and test true positive links
    split = int(len(row) * (1 - test_ratio - train_ratio))
    adj_train_edges = (row[:split], col[:split])
    rest_edges = (row[split:], col[split:])

    adj_train_values = np.ones(len(adj_train_edges[0]))
    adj_train = csr_matrix( (adj_train_values, adj_train_edges), shape=(num_nodes, num_nodes) )

    # insert negative edges
    num_to_insert = len(rest_edges[0])

    # TODO: add check that there are actually num_to_insert negative edges to insert
    # we should insert them checking directly where there is 0 in the adj matrix (but not in cells that refer to gene-gene links or disease-disease links)

    neg_edges = set()
    while len(neg_edges) < num_to_insert:

        i,j = random.randint(0, num_genes-1), random.randint(num_genes, num_nodes-1)
        if adj[i, j] == 0:
            neg_edges.add( (i,j) )
        else:
            continue

    row,col = [],[]
    for elem in neg_edges:
        row.append(elem[0])
        col.append(elem[1])
    
    neg_edges = ( np.array(row), np.array(col) )
    
    # now split the rest_set in two sets: 
    # - the set we will use to train 
    # - the set we will use to test

    # split2 between res (train and validation) and test edges
    split2 = int( len(rest_edges[0]) * (1- (test_ratio/(test_ratio+train_ratio))) )
    res_pos_edges = (rest_edges[0][:split2], rest_edges[1][:split2])
    test_pos_edges = (rest_edges[0][split2:], rest_edges[1][split2:])
    
    res_neg_edges = (neg_edges[0][:split2], neg_edges[1][:split2])
    test_neg_edges = (neg_edges[0][split2:], neg_edges[1][split2:])
    
    assert(len(res_pos_edges[0]) == len(res_neg_edges[0])), 'Wrong number of negative edges added!'
    assert(len(test_pos_edges[0]) == len(test_neg_edges[0])), 'Wrong number of negative edges added!'

    # split3: between validation and train edges
    split3 = int( len(res_pos_edges[0]) * (1- (test_ratio/(test_ratio+train_ratio))) )
    train_pos_edges = (res_pos_edges[0][:split3], res_pos_edges[1][:split3])
    validation_pos_edges = (res_pos_edges[0][split3:], res_pos_edges[1][split3:])

    train_neg_edges = (res_neg_edges[0][:split3], res_neg_edges[1][:split3])
    validation_neg_edges = (res_neg_edges[0][split3:], res_neg_edges[1][split3:])

    # hstack of train, test and validation edges
    train_edges = ( np.hstack((train_pos_edges[0], train_neg_edges[0])), np.hstack((train_pos_edges[1], train_neg_edges[1])) )
    test_edges = ( np.hstack((test_pos_edges[0], test_neg_edges[0])), np.hstack((test_pos_edges[1], test_neg_edges[1])) )
    validation_edges = ( np.hstack((validation_pos_edges[0], validation_neg_edges[0])), np.hstack((validation_pos_edges[1], validation_neg_edges[1])) )

    # labels definition
    train_labels = np.hstack( (np.ones(len(train_pos_edges[0])), np.zeros(len(train_neg_edges[0]))) )
    test_labels = np.hstack( (np.ones(len(test_pos_edges[0])),  np.zeros(len(test_neg_edges[0]))) )
    validation_labels = np.hstack( (np.ones(len(validation_pos_edges[0])),  np.zeros(len(validation_neg_edges[0]))) )

    # permutaion of the three subsets
    perm_train = np.random.permutation(range(len(train_labels)))
    perm_test = np.random.permutation(range(len(test_labels)))
    perm_valid = np.random.permutation(range(len(validation_labels)))


    train_edges = (train_edges[0][perm_train], train_edges[1][perm_train])
    train_labels = train_labels[perm_train]

    test_edges = (test_edges[0][perm_test], test_edges[1][perm_test])
    test_labels = test_labels[perm_test]
    
    validation_edges = (validation_edges[0][perm_valid], validation_edges[1][perm_valid])
    validation_labels = validation_labels[perm_valid]

    return adj_train, train_edges, train_labels, validation_edges, validation_labels, test_edges, test_labels

def create_adj_matrix(data, num_nodes, nodes_dict):
    # creating an adjacency matrix with the genes on the rows and the diseases on the columns

    values = np.ones(2*len(data))

    row = []
    col = []
    for idx in range(len(data)):
        gene = data['genes'][idx]
        disease = data['diseases'][idx]

        gene_idx = nodes_dict[gene]
        disease_idx = nodes_dict[disease]
        
        row.append(gene_idx)
        col.append(disease_idx)

        row.append(disease_idx)
        col.append(gene_idx)

    row = np.array(row)
    col = np.array(col)

    adj = csr_matrix( (values, (row, col)), shape = (num_nodes, num_nodes) )
    return adj

def save_to_file(row, col, fname, labels=None):
    with open(fname, 'w') as fout:
        for i in range(len(col)):
            if(labels is None):
                fout.write('{}\t{}\n'.format(row[i], col[i]))
            else:
                fout.write('{}\t{}\t{}\n'.format(row[i], col[i], labels[i]))


# this function first turns the label from strings to integers (in order to 
# meet the requirements of networkx) than splits tha data into train, test and validation
# and saves these files, so that we do not have to recompute them. 
def split_dataset():
		
    data = pd.read_csv(constants.DATASET_FILE_NAME, sep='\t', header=0, names=['diseases','genes'])

    diseases = sorted(set(data['diseases']))
    genes = sorted(set(data['genes']))

    num_diseases = len(diseases)
    num_genes = len(genes)
    num_nodes = num_diseases + num_genes

    genes_dict = {genes[idx]:idx for idx in range(num_genes)}
    diseases_dict = {diseases[idx]:idx for idx in range(num_diseases)}

    nodes_dict = genes_dict.copy()
    for key in diseases_dict:
        nodes_dict[key] = diseases_dict[key]+num_genes

    adj = create_adj_matrix(data, num_nodes, nodes_dict)

    adj_train, train_edges, train_labels, validation_edges, validation_labels, test_edges, test_labels = train_test_split(adj, num_genes)

    # adj_train: adj to use to compute node2vec
    # get upper triangular matrix (without the main diagonal, so k=1, otherwise 0)
    adj_triu = sp.triu(adj_train, k=1)
    
    # return row and col indices of non-zero values in net_triu
    row, col, _ = sp.find(adj_triu)

    save_to_file(row, col, constants.ADJ_FILE_NAME)
    save_to_file(train_edges[0], train_edges[1], constants.TRAIN_EDGES_FILE_NAME,  labels=train_labels)
    save_to_file(test_edges[0], test_edges[1], constants.TEST_EDGES_FILE_NAME,  labels=test_labels)
    save_to_file(validation_edges[0], validation_edges[1], constants.VALIDATION_EDGES_FILE_NAME,  labels=validation_labels)

    