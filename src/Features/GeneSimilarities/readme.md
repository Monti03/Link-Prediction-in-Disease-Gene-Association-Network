To run this code in order to get the features of the genes it should be present a folder in this directory named *data* within which the dataset should be present (obtainable from this [link](https://snap.stanford.edu/biodata/datasets/10020/10020-DG-Miner.html)) under this name: *DG-Miner_miner-disease-gene.tsv*. So, in addition to the dataset file, other files will be produced when running this code:

* *./GO_terms_dict.txt*: dictionary containing the mapping between the codes of the genes used in the dataset and the *GO* terms associated to them.
* *./data/go-basic.obo*: DAG containing the GO terms ontology.
* *./genes_features_matrix.npy*: file containing the feature matrix of genes (similarity matrix).

