import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.neighbors
import sklearn.linear_model

def knn_evaluate(data_train, labels_train, data_test, labels_test, n_neighbors):
    neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', algorithm='auto')
    neigh.fit(data_train, labels_train)
    predict_test = neigh.predict(data_test)
    return float(np.sum(predict_test == labels_test)) / len(labels_test)

def logreg_evaluate(data_train, labels_train, data_test, labels_test):
    logreg = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
    logreg.fit(data_train, labels_train)
    predict_test = logreg.predict(data_test)
    return float(np.sum(predict_test == labels_test)) / len(labels_test)

def main():
    prefix, = sys.argv[1:]
    npz = np.load(prefix+".npz") # leaking file descriptor, but whatever
    keys = npz.keys()
    layer_count = len([key for key in keys if key.startswith("train-")])

    with np.load(prefix+"-labels.npz") as npz_labels:
        labels_train = npz_labels["train"]
        labels_test = npz_labels["test"]

    for layer_index in range(layer_count):
        data_train = npz["train-"+str(layer_index)]
        data_test  = npz["test-"+str(layer_index)]
        # batch_index//20 x sample_index x neuron_index
        print "taking first row of data_train, not sure what it means."
        data_train = data_train[0]
        data_test = data_test[0]
        labels_train = labels_train[:len(data_train)]
        labels_test  = labels_test [:len(data_test )]
        # data_train = data_train.reshape(-1, data_train.shape[-1])
        # data_test  = data_test .reshape(-1, data_test .shape[-1])
        for n_neighbors in range(1, 8):
            knn_accuracy = knn_evaluate(data_train, labels_train, data_test, labels_test, n_neighbors=n_neighbors)
            print "%d-nn accuracy at layer %d: %f" % (n_neighbors, layer_index, knn_accuracy)
        logreg_accuracy = logreg_evaluate(data_train, labels_train, data_test, labels_test)
        print "multinomial logistic regression accuracy at layer %d: %f" % (layer_index, logreg_accuracy)


main()
