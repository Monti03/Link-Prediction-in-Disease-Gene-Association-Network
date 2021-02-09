
File description

- *data_description.json* -> raw obtained descriptions
- *null_description.txt* -> identifiers of the diseases with no description in the precedent file
- *null_description_page.txt* -> identifiers of the diseases for which there was no page 
- *data_description_updated* -> updated descriptions: removed the diseases with no valid description and replaced the --description of the diseaseIDs related to multiple diseases 
- *disease_ids_order.txt* -> order I will consider
- *normalized_texts.txt* -> for each row we have the normalized texts (considering the precedent order)
- *tf_idf_data.npz* -> contains the tf_idf matrix 
- *tf_idf_data_SVD* -> contains the tf_idf matrix after SVD

In order to run the notebook you have to add the [dataset](https://snap.stanford.edu/biodata/datasets/10020/10020-DG-Miner.html) in this folder as `DG-Miner_miner-disease-gene.tsv`