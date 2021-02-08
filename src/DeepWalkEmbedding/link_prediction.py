from gensim.models import KeyedVectors

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

import consts as constants

def load_n2v_embedding():
    wv = KeyedVectors.load('word2vec.wordvectors', mmap='r')
    return wv

def build_link_embedding(node1, node2, wv):
    if(node1 in wv and node2 in wv):
        e_1 = wv[node1]
        e_2 = wv[node2]

        return np.concatenate((e_1,e_2))
    else:
        # can happen that some nodes with few links do not compare in the 
        # subset of edges used for building the embedding, so these edges
        # have no embedding, so I canno use them. 
        return None

# return a np array for each edge where the first 100 values
# represent the disease and the last 100 values represent the 
# gene. to_remove is a set of indexes of edges to remove since 
# at least one of the two relative nodes have no embedding
def build_link_embeddings(edges, wv):
    ret = []
    to_remove = set()
    for i in range(len(edges)):
        edge = edges[i]
        value = build_link_embedding(edge[0], edge[1], wv)
        if(not value is None):
            ret.append(value)
        else:
            to_remove.add(i)
    return np.array(ret), to_remove


def load_data():
    train_edges, test_edges, validation_edges = [], [], []
    train_labels, test_labels, validation_labels = [], [], []
    with open(constants.TRAIN_EDGES_FILE_NAME) as fin:
        for line in fin:
            if line == '':
                break
            train_edges.append((line.split('\t')[0], line.split('\t')[1]))
            train_labels.append(float(line.split('\t')[2].strip()))

    with open(constants.TEST_EDGES_FILE_NAME) as fin:
        for line in fin:
            if line == '':
                break
            test_edges.append((line.split('\t')[0], line.split('\t')[1]))
            test_labels.append(float(line.split('\t')[2].strip()))
    
    with open(constants.VALIDATION_EDGES_FILE_NAME) as fin:
        for line in fin:
            if line == '':
                break
            validation_edges.append((line.split('\t')[0], line.split('\t')[1]))
            validation_labels.append(float(line.split('\t')[2].strip()))

    return (train_edges, train_labels), (test_edges, test_labels), (validation_edges, validation_labels)

def remove_indexes(to_remove_indexes, list_):
    
    for index in sorted(to_remove_indexes, reverse=True):
        del list_[index]
    

def plot(history):
    #print(history.history.keys())
    fig, ax = plt.subplots()
    # accuracy
    ax.plot(range(len(history.history['accuracy'])), history.history['accuracy'], label='train_accuracy')
    ax.plot(range(len(history.history['val_accuracy'])), history.history['val_accuracy'],  label='valdiation_accuracy')
    ax.legend(['train_accuracy', 'validation_accuracy'])
    fig.savefig('accuracy.png')

    fig, ax = plt.subplots()
    # loss
    ax.plot(range(len(history.history['loss'])), history.history['loss'], label='train_loss')
    ax.plot(range(len(history.history['val_loss'])), history.history['val_loss'], label='valdiation_loss')
    ax.legend(['train_loss', 'validation_loss'])
    fig.savefig('loss.png')

def main():

    wv = load_n2v_embedding()

    train, test, validation = load_data()

    train_edges, train_to_remove = build_link_embeddings(train[0], wv)
    test_edges, test_to_remove = build_link_embeddings(test[0], wv)
    validation_edges, validation_to_remove = build_link_embeddings(validation[0], wv)

    remove_indexes(list(train_to_remove), train[1])
    remove_indexes(list(test_to_remove), test[1])
    remove_indexes(list(validation_to_remove), validation[1])

    
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid') 
    ])

    model.compile(
        optimizer='adam', loss=loss_fn, metrics=['accuracy']
    )
    
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, mode='min'
    )

    history = model.fit(train_edges, np.array(train[1]), epochs=500, 
        validation_data=(validation_edges, np.array(validation[1])),
        batch_size=constants.BATCH_SIZE, callbacks=[es]
    )

    model.evaluate(test_edges,  np.array(test[1]), verbose=2)

    plot(history)

if __name__ == '__main__':
    main()
