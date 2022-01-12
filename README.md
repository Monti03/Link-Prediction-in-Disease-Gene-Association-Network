# Link Prediction in Disease Gene Association Network
## Motivation
The information about genes and variants involved in human diseases can be used for different research
purposes, including the investigation of molecular mechanisms of species diseases and their
comorbidities, the analysis of the properties of disease genes, the generation of hypotheses on drug
therapeutic action and drug adverse effects, the validation of computationally predicted disease genes,
and the evaluation of text-mining methods performance.

This project focuses on the task of link prediction on disease-genes association network, which are such
that nodes are genes and diseases and edges represent associations between them.

## Dataset
The dataset  we used is the disease-genes association network you can find at [link](https://snap.stanford.edu/biodata/datasets/10020/10020-DG-Miner.html)

<table>
      <tr><td>Nodes</td><td>23484</td></tr>
      <tr><td>Disease nodes </td><td>5663</td></tr>
      <tr><td> Gene nodes </td><td>17821</td></tr>
      <tr><td>Edges </td><td>15509618</td></tr>
</table>

## Features
In order to have our models produce better results, we collected some gene and disease features to be used along the adjacency matrix of the graph.

We considered as Features of both genes and diseases their respective similarity. In particular, the genes feature matrix contains gene-gene similarities and the diseases feature matrix contains disease-disease similarities.

### Gene Features
Each gene is typically associated to a set of GO (Gene Ontology) terms; each of these terms refers to some specific characteristic of genes. GO terms are organized in DAGs (Directed Acyclic Graphs) which define their ontology; we can have three types of ontologies: Molecular Function, Biological Process and Cellular Component. We picked Molecular Function’s DAG and computed the similarity between every pair of genes that are present in our dataset considering them as sets of GO terms and implementing a similarity between such sets.

### Disease Features
The diseases can be represented as sets of terms. In this case, each set contains DO (Disease Ontology) terms that represent the causes of the disease. The similarity measure we used is the same for GO terms and DO terms and is in the following slide.

## Link Prediction Methods
We tried three different approaches: Weisfeiler-Lehman Link Prediction, DeepWalk and Graph Convolutional Networks

### Weisfeiler-Lehman Link Prediction
This model consists in extracting various subgraphs from the original graph and use them to predict the existence of links. In order to make such prediction we: 
- compute an embedding for each link,
- predict if a link is present or not.

#### Link Embeddings 
To compute the embedding of a link we compute the subgraph associated to it: each subgraph is the flattened vector associated to the strictly upper triangular adjacency matrix relative to the subgraph (after having removed the entry that codifies the presence of the link of interest). Moreover, in order to get vectors that are meaningful and coherent with each other, we determined the number of nodes for each subgraph to be equal to 10 and sorted all the nodes that appear in each subgraph according to the same metric: vertices receive similar rankings if their relative positions and structural roles within their respective subgraphs are also similar. 

<p align="center" width="100%">
    <img src="https://user-images.githubusercontent.com/38753416/149150333-2e3b6b58-d1c9-4081-8ce4-797d0bfb4a0a.png">
</p>

#### Link Prediction
In order to predict the presence (or not) of a link we use a three layer feedforward fully connected neural network. The input to the network is the embedding of a link, which is a vector of 44 entries. The output of the network is a value that represents a degree of confidence with which the input link it’s believed to exist.


### DeepWalk
With this approach we first build the embedding of the nodes of the graph by using DeepWalk technique and then use them to predict the existence of links.  In order to make such prediction we:
- compute the embedding for each node,
- compute the embedding of each link,
- predict if a link is present or not.

#### Node Embedding
We used DeepWalk in order to build the embedding of each node of the network. This method is divided in two phases:
- generate truncated random walks
- use word2vec in order to generate the embedding 

The truncated random walks are represented as a sequence of nodes. These sequences can be thought of short sentences in a special language, so we can apply word2vec (the skipgram version) to build the embedding of each node (term of the sentence).

In this implementation I decided to use 50 walks starting from each node, each one of length 30. In
word2vec I used a window size of 3 and the resulting embedding of the nodes is of size 100.

#### Link Prediction
In order to perform the task of link prediction, we compute a link embedding. Each link embedding will be defined as the concatenation of the embeddings of the two nodes which participate to the link (the first is the gene, the second is the disease).

In order to predict the presence of a link we use a simple fully connected neural network.

## Graph Convolutional Neural Networks
With this approach we use a Graph Convolutional Neural Network (GCN) to build the embedding of each node. The embeddings of the nodes are then used to predict the existence or not of the link to which the nodes would participate.

The neural network has the following structure:
- a graph convolutional block with 64 nodes,
- a ReLU activation function,
- a graph convolutional block with 64 nodes.

#### Convolutional Block
Each Convolutional Block computes the output of the layer as:

<p align="center" width="100%">
    <img src="http://www.sciweavers.org/tex2img.php?eq=H%5E%7B%28l%2B1%29%7D%20%3D%20%5Csigma%28%7BD%7D%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D%20%5Ctilde%7BA%7D%20%7BD%7D%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D%20H%5E%7B%28l%29%7D%20W%5E%7B%28l%29%7D%29%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0">
</p>

Where Ã is the adjacency matrix after self loops on every node have been added, D is the degree matrix, which is a diagonal matrix whose element at row i is the degree of node i. H<sup>(l)</sup> s the output of layer l-1 (in
the first layer this matrix is represented by the features of nodes) and W<sup>(l)</sup> is the weight matrix at layer l

#### Link Prediction
The link prediction task is performed in the following way:
- compute the scalar product of the embedding of two nodes (a gene and a disease),
- apply sigmoid activation function to the obtained value,
- round the value to the nearest integer

## Results
|Method| Accuracy| F1| Prec| Rec |
|---| ---| ---| ---| --- |
|Weisfeiler-Lehman|  0.57| 0.350| 0.724| 0.231 |
|DeepWalk| 0.95 | 0.95 | 0.95| 0.95 |
|GCN| 0.95| 0.95| 0.95| 0.96 |

## Conclusions
According to accuracy metric the technique employing Graph Convolutional Networks and the one which uses DeepWalk embeddings are the best ones. The memory and time constraints we encountered when training the GCN, though, have been stricter: we were not able to train it using a GPU because of high memory usage and as a result each epoch took something like 12 minutes, resulting in almost 50 hours of training. The DeepWalk based method, instead, required 2 hours to build the embeddings and 45 minutes to train the model, with less strict memory requirements.

