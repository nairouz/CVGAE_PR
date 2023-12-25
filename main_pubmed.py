#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authors : Nairouz Mrabah (mrabah.nairouz@courrier.uqam.ca) 
# @License : MIT License

import numpy as np
import torch
import scipy.sparse as sp
import matplotlib.pyplot as plt
import math
import itertools
from model_pubmed import CVGAE
from preprocessing import load_data, mask_test_edges, sparse_to_tuple, preprocess_graph
from sklearn.metrics import confusion_matrix

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def map_vector_to_clusters(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_true[i], y_pred[i]] += 1
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    y_true_mapped = np.zeros(y_pred.shape)
    for i in range(y_pred.shape[0]):
        y_true_mapped[i] = col_ind[y_true[i]]
    return y_true_mapped.astype(int)

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.1f}".format(round_half_up(cm[i, j], decimals=1)),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label \n accuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig("./CVGAE/results/Pubmed/train/confusion_matrix.png")

# Dataset Name
dataset = "Pubmed"
print("Pubmed dataset")
adj, features, labels = load_data('pubmed', './CVGAE/data/Pubmed')
nClusters = 3

# Network parameters
alpha = 1.
gamma_1 = 1.
gamma_2 = 1.
gamma_3 = 1.
num_neurons = 32
embedding_size = 16
save_path = "./CVGAE/results/"

# Some preprocessing
adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj.eliminate_zeros()
#adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
#adj = adj_train
adj_norm = preprocess_graph(adj)
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
adj_label = adj + sp.eye(adj.shape[0])
adj_label = sparse_to_tuple(adj_label)
adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]), torch.Size(adj_norm[2])).to("cuda:4")
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]), torch.Size(adj_label[2])).to("cuda:4")
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), torch.FloatTensor(features[1]), torch.Size(features[2])).to("cuda:4")
weight_mask_orig = adj_label.to_dense().view(-1) == 1
weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
weight_tensor_orig[weight_mask_orig] = pos_weight_orig
weight_tensor_orig = weight_tensor_orig.to("cuda:4")

# Create and train Model
plt.figure()
network = CVGAE(num_neurons=num_neurons, num_features=num_features, embedding_size=embedding_size, nClusters=nClusters, activation="ReLU", alpha=alpha, gamma_1=gamma_1, gamma_2=gamma_2, gamma_3=gamma_3).to("cuda:4")
#y_pred, y = network.pretrain(features, adj_norm, adj_label, labels, weight_tensor_orig, norm, optimizer="Adam", epochs=6000, lr=0.001, save_path=save_path, dataset=dataset)
y_pred, y = network.train(features, adj_norm, adj_label, adj, labels, weight_tensor_orig, norm, optimizer="Adam", epochs=1000, lr=0.001, beta_1=0., beta_2=0.32, save_path=save_path, dataset=dataset)

target_names = ["0", "1", "2"]
y_mapped = map_vector_to_clusters(y, y_pred)
cm = confusion_matrix(y_true=y_mapped, y_pred=y_pred, normalize='true')
plot_confusion_matrix(cm, target_names, title='Confusion matrix', normalize=True, cmap=plt.cm.Blues)
