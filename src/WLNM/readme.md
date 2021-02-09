To run this notebook in order to apply the *WLNM* method for link prediction the dataset (obtainable from this [link](https://snap.stanford.edu/biodata/datasets/10020/10020-DG-Miner.html)) should be present in the following relative path with respect to the current directory with the following name: *./data/DG-Miner_miner-disease-gene.tsv*. The execution of this notebook will produce two files (which are now already present in the repo):

* *X_train.npy* : it contains the embeddings of the training links, in an order consistent with the training labels present in the code.
* *X_test.npy* : it contains the embeddings of the test links, in an order consistent with the test labels present in the code.

The implementation of this notebook refers to the Weisfeiler-Lehman Neural Machine (WLNM) method [[1]](#1) (the paper is present in the directory *precedentWorks*).



## References
<a id="1">[1]</a>  Muhan Zhang and Yixin Chen. Weisfeiler-lehman neural machine for link prediction. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 575–583. ACM, 2017.