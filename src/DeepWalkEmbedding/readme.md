
- main.py creates the embedding of the nodes. This file uses also:
  - graph.py
  - RandomWalkThread.py (that is used in order to parallelize the creation of the random walks)
  - utils.py that is used in order to preprocess the data (like split the data)
- link_prediction.py trains a nn for link prediction

in order to run main.py you have to add a folder named `data` containing the dataset that you can find at the following [link](https://snap.stanford.edu/biodata/datasets/10020/10020-DG-Miner.html)
