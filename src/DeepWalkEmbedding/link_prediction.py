from gensim.models import KeyedVectors

import tensorflow as tf
import numpy as np

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve
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
    print(history.history.keys())
    fig, ax = plt.subplots()
    # accuracy
    ax.plot(range(len(history.history['acc'])), history.history['acc'], label='train_accuracy')
    ax.plot(range(len(history.history['val_acc'])), history.history['val_acc'],  label='valdiation_accuracy')
    ax.legend(['train_accuracy', 'validation_accuracy'])
    fig.savefig('accuracy.png')

    fig, ax = plt.subplots()
    # loss
    ax.plot(range(len(history.history['loss'])), history.history['loss'], label='train_loss')
    ax.plot(range(len(history.history['val_loss'])), history.history['val_loss'], label='valdiation_loss')
    ax.legend(['train_loss', 'validation_loss'])
    fig.savefig('loss.png')

def roc_curve_plot(testy, y_pred):
    lr_auc = roc_auc_score(testy, y_pred)

    print('Logistic: ROC AUC=%.3f' % (lr_auc))

    lr_fpr, lr_tpr, _ = roc_curve(testy, y_pred)
    
    plt.plot(lr_fpr, lr_tpr, label='ROC AUC=%.3f' % (lr_auc))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend()
    plt.savefig('ROC_score.png')
    plt.show()


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

    test_pos = np.sum(test[1])
    test_neg = np.sum(1-np.array(test[1]))
    print('pos:{}'.format(test_pos))
    print('neg:{}'.format(test_neg))

    indices = [i for i, x in enumerate(test[1]) if x == 1]
    indices = indices[:int(test_pos -test_neg)]
    remove_indexes(indices, test[1])

    test_edges = np.delete(test_edges, indices, 0)
    test_pos = np.sum(test[1])
    test_neg = np.sum(1-np.array(test[1]))
    print('post pos:{}'.format(test_pos))
    print('post neg:{}'.format(test_neg))

    print("-------------------------")
    print(train_edges)
    print("-------------------------")

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.1), 
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid') 
    ])

    model.compile(
        optimizer='adam', loss=loss_fn, metrics=['accuracy']
    )
    
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, mode='min'
    )

    history = model.fit(train_edges, np.array(train[1]), epochs=500, 
        validation_data=(validation_edges, np.array(validation[1])),
        batch_size=constants.BATCH_SIZE, callbacks=[es]
    )

    model.evaluate(test_edges,  np.array(test[1]), verbose=2)

    predictions = model.predict(test_edges)

    roc_curve_plot(np.array(test[1]), predictions)


    predictions = (predictions > 0.5)
    cm = confusion_matrix(np.array(test[1]), predictions)
    
    df_cm = pd.DataFrame(cm, index = [i for i in "01"],
                  columns = [i for i in "01"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='g')

    plt.savefig('confusion_matrix.png')
    plt.show()
    plot(history)
    print()
    print(classification_report(np.array(test[1]), predictions))
    
    diff = np.array(test[1])-predictions.flatten()
    print()
    print('Correctly classified: ', np.where(diff == 0)[0].shape[0])
    print('Incorrectly classified: ', np.where(diff != 0)[0].shape[0])
    print('False positives: ', np.where(diff == -1)[0].shape[0])
    print('False negatives: ', np.where(diff == 1)[0].shape[0])
    print()

    model.save('model')

if __name__ == '__main__':
    main()
