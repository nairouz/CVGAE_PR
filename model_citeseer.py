#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authors : Nairouz Mrabah (mrabah.nairouz@courrier.uqam.ca) 
# @Link    : github.com/nairouz/CVGAE
# @Paper   : Beyond the Evidence Lower Bound: A Contrastive Variatonal Graph Auto-Encoder for Attributed Graph Clustering
# @License : MIT License

import os
import torch
import csv
import sklearn
import numpy as np
import torch.nn as nn
import seaborn as sns
import metrics as mt
import networkx as nx
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.sparse as sp
from torch.nn import Parameter
from tqdm import tqdm
from torch.optim import Adam, SGD, RMSprop
from sklearn import metrics
from torch.optim.lr_scheduler import StepLR
from sklearn.manifold import TSNE
from munkres import Munkres
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from preprocessing import sparse_to_tuple, preprocess_graph
from sklearn.neighbors import NearestNeighbors

class Clustering_Metrics:
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]
            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        ari = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        print('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ARI_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, ari))

        fh = open('recoder.txt', 'a')

        fh.write('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ARI_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, ari) )
        fh.write('\r\n')
        fh.flush()
        fh.close()

        return acc, nmi, ari, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

class ClusterAssignment(nn.Module):
    def __init__(self, cluster_number, embedding_dimension, alpha, cluster_centers=None):
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.cluster_number, self.embedding_dimension, dtype=torch.float)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, inputs):
        norm_squared = torch.sum((inputs.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = random_uniform_init(input_dim, output_dim) 
        self.activation = activation
        
    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)        
        outputs = self.activation(x)
        return outputs

def purity_score(y_true, y_pred):
    contingency_matrix = sklearn.metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def random_uniform_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

def generate_unconflicted_data_index(p, beta_1, beta_2):
    unconf_indices = []
    conf_indices = []
    p = p.detach().cpu().numpy()
    confidence1 = p.max(1)
    confidence2 = np.zeros((p.shape[0],))
    a = np.argsort(p, axis=1)[:,-2]
    for i in range(p.shape[0]):
        confidence2[i] = p[i,a[i]]
        if (confidence1[i] > beta_1) and (confidence1[i] - confidence2[i]) > beta_2:
            unconf_indices.append(i)
        else:
            conf_indices.append(i)
    unconf_indices = np.asarray(unconf_indices, dtype=int)
    conf_indices = np.asarray(conf_indices, dtype=int)
    return unconf_indices, conf_indices

def generate_sep_index(emb, centers, p):
    emb = emb.detach().cpu().numpy()
    centers = centers.detach().cpu().numpy()
    p = p.detach().cpu().numpy()
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(emb)
    _, indices = nn.kneighbors(centers)
    indices_sep = np.zeros((emb.shape[0], centers.shape[0]-2), dtype=int)
    assignments_index = np.argsort(p, axis=1)
    first_center_index = indices[assignments_index[:,-1]]
    second_center_index = indices[assignments_index[:,-2]]
    for i in range(emb.shape[0]):
        k = 0
        for j in indices:
            if (j != first_center_index[i]) and (j != second_center_index[i]):
                indices_sep[i, k] = j
                k+=1
    return indices_sep

def negative_embeddings(z_mu_pos, z_sigma2_log_pos, emb_pos):
    idx = torch.randperm(emb_pos.shape[0])
    z_mu_neg = z_mu_pos[idx,:]
    z_sigma2_log_neg = z_sigma2_log_pos[idx,:]
    emb_neg = emb_pos[idx,:]
    return z_mu_neg, z_sigma2_log_neg, emb_neg

def target_distribution(p, unconflicted_ind, conflicted_ind):
    p = p.detach().cpu().numpy()
    q = np.zeros(p.shape)
    q[conflicted_ind] = p[conflicted_ind]
    q[unconflicted_ind, np.argmax(p[unconflicted_ind], axis=1)] = 1
    q = torch.tensor(q, dtype=torch.float32).to("cuda:4")
    return q

def evaluate_links(adj, labels):
    count_links = {"nb_links": 0,
                   "nb_false_links": 0,
                   "nb_true_links": 0}
    for i, line in enumerate(adj):
        for j in range(line.indices.size):
            if line.indices[j] > i:
                count_links["nb_links"] += 1
                if labels[i] == labels[line.indices[j]]:
                    count_links["nb_true_links"] += 1
                else:
                    count_links["nb_false_links"] += 1
    return count_links

class CVGAE(nn.Module):

    def __init__(self, **kwargs):
        super(CVGAE, self).__init__()
        self.num_neurons = kwargs['num_neurons']
        self.num_features = kwargs['num_features']
        self.embedding_size = kwargs['embedding_size']
        self.nClusters = kwargs['nClusters']
        if kwargs['activation'] == "ReLU":
            self.activation = F.relu
        if kwargs['activation'] == "Sigmoid":
            self.activation = F.sigmoid
        if kwargs['activation'] == "Tanh":
            self.activation = F.tanh
        self.alpha = kwargs['alpha']
        self.gamma_1 = kwargs['gamma_1']
        self.gamma_2 = kwargs['gamma_2']
        self.gamma_3 = kwargs['gamma_3']

        # VGAE training parameters
        self.base_gcn = GraphConvSparse(self.num_features, self.num_neurons, self.activation)
        self.gcn_mean = GraphConvSparse(self.num_neurons, self.embedding_size, activation=lambda x:x)
        self.gcn_logsigma2 = GraphConvSparse(self.num_neurons, self.embedding_size, activation=lambda x:x)
        self.assignment_1 = ClusterAssignment(self.nClusters, self.embedding_size, self.alpha)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean') 

    def generate_centers(self, emb_unconf):
        y_pred = self.predict(emb_unconf)
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(emb_unconf.detach().cpu().numpy())
        _, indices = nn.kneighbors(self.assignment_1.cluster_centers.detach().cpu().numpy())
        return indices[y_pred] 

    def update_graph(self, adj, emb, labels, unconf_indices):
        count_target_links = {"nb_added_links": 0,
                              "nb_false_added_links": 0,
                              "nb_true_added_links": 0,
                              "nb_deleted_links": 0,
                              "nb_false_deleted_links": 0,
                              "nb_true_deleted_links": 0}
        y_pred = self.predict(emb)
        emb_unconf = emb[unconf_indices]
        adj_pos = adj.tolil()
        idx = unconf_indices[self.generate_centers(emb_unconf)]    
        for i, k in enumerate(unconf_indices):
            adj_k_pos = adj_pos[k].tocsr().indices
            if not(np.isin(idx[i], adj_k_pos)) and (y_pred[k] == y_pred[idx[i]]):
                if labels[k] == labels[idx[i]]:
                    count_target_links["nb_true_added_links"] += 1
                else:
                    count_target_links["nb_false_added_links"] += 1
                count_target_links["nb_added_links"] += 1
                adj_pos[k, idx[i]] = 1
            for j in adj_k_pos:
                if np.isin(j, unconf_indices) and (np.isin(idx[i], adj_k_pos)) and (y_pred[k] != y_pred[j]):
                    if labels[k] == labels[j]:
                        count_target_links["nb_true_deleted_links"] += 1
                    else:
                        count_target_links["nb_false_deleted_links"] += 1
                    count_target_links["nb_deleted_links"] += 1
                    adj_pos[k, j] = 0
        adj_pos = adj_pos - sp.dia_matrix((adj_pos.diagonal()[np.newaxis, :], [0]), shape=adj_pos.shape)
        adj_pos = adj_pos.tocsr()
        adj_pos.eliminate_zeros()
        adj_norm_pos = preprocess_graph(adj_pos)
        pos_weight = float(adj_pos.shape[0] * adj_pos.shape[0] - adj_pos.sum()) / adj_pos.sum()
        norm_pos = adj_pos.shape[0] * adj_pos.shape[0] / float((adj_pos.shape[0] * adj_pos.shape[0] - adj_pos.sum()) * 2)
        adj_label_pos = adj_pos + sp.eye(adj_pos.shape[0])
        adj_label_pos = sparse_to_tuple(adj_label_pos)
        adj_norm_pos = torch.sparse.FloatTensor(torch.LongTensor(adj_norm_pos[0].T), torch.FloatTensor(adj_norm_pos[1]), torch.Size(adj_norm_pos[2])).to("cuda:4")
        adj_label_pos = torch.sparse.FloatTensor(torch.LongTensor(adj_label_pos[0].T), torch.FloatTensor(adj_label_pos[1]), torch.Size(adj_label_pos[2])).to("cuda:4")
        weight_mask_pos = adj_label_pos.to_dense().view(-1) == 1
        weight_tensor_pos = torch.ones(weight_mask_pos.size(0))
        weight_tensor_pos[weight_mask_pos] = pos_weight
        weight_tensor_pos = weight_tensor_pos.to("cuda:4")
        return adj_pos, adj_norm_pos, adj_label_pos, weight_tensor_pos, norm_pos, count_target_links

    @staticmethod
    def update_features(features):
        features_dense = features.to_dense()
        idx = np.random.permutation(features_dense.shape[0])
        features_neg = features_dense[idx,:]
        indices = torch.nonzero(features_neg).t()
        values = features_neg[indices[0], indices[1]] 
        features_neg = torch.sparse.FloatTensor(indices, values, features_neg.size())
        return features_neg

    def compute_separtion_loss(self, z_mu_pos, z_sigma2_log_pos, sep_ind, unconflicted_ind):
        # Preparing negative signals
        z_mu_neg_tensor = z_mu_pos[sep_ind,:][unconflicted_ind]
        z_mu_pos_tensor = z_mu_pos[unconflicted_ind].unsqueeze(1).repeat_interleave(self.nClusters-2, dim=1)

        # Computing the separation loss
        KL_neg = (1 / z_mu_pos.shape[0]) * torch.einsum("ijk,ijk->ij", (z_mu_pos_tensor - z_mu_neg_tensor), (z_mu_pos_tensor - z_mu_neg_tensor))
        Loss_sep = torch.mean(torch.log(1 + torch.mean(torch.exp(-KL_neg), dim=1)), dim=0)
        return Loss_sep

    def pretrain(self, features, adj_norm, adj_label, y, weight_tensor, norm, optimizer="Adam", epochs=200, lr=0.01, save_path="/home/mrabah_n/code/CVGAE/results/", dataset="Cora"):
        if optimizer == "SGD":
            opti = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay = 0.001)
        elif optimizer == "RMSProp":
            opti = RMSprop(self.parameters(), lr=lr, weight_decay=0.001)
        else:
            opti = Adam(self.parameters(), lr=lr)

        logfile = open(save_path + dataset + '/pretrain/log_train.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'pur', 'f1_macro', 'f1_micro', 'precision_macro', 'precision_micro', 'recall_macro', 'recall_micro', 'loss'])
        logwriter.writeheader()
        km = KMeans(n_clusters=self.nClusters, n_init=20)

        ##############################################################
        # Training loop
        print("")
        print("Training......")
        epoch_bar = tqdm(range(epochs))
        acc_best = 0
        y_pred_best = 0
        for epoch in epoch_bar:
            opti.zero_grad()
            z_mu, z_sigma2_log, z, hidden = self.encode(features, adj_norm)
            adj_out = self.decode(z)
            adj_out_np = adj_out.detach().cpu().numpy()
            index = np.where(adj_out_np > 0.5)
            adj_out_csr = csr_matrix((np.ones((index[0].shape)), index), shape=adj_out_np.shape, dtype=int)
            Loss_recons = norm * F.binary_cross_entropy(adj_out.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor) 
            Loss_reg = - (0.5 / adj_out.size(0)) * (1 + z_sigma2_log - z_mu**2 - torch.exp(z_sigma2_log)).sum(1).mean()
            loss = Loss_recons + Loss_reg
            loss.backward()
            opti.step()

            ##############################################################
            # Evaluation
            epoch_bar.write('Loss pretraining = {:.4f}'.format(loss))
            print("")
            print("loss reconstruction: " + str(Loss_recons.detach().cpu().numpy()))
            print("loss regularization: " + str(Loss_reg.detach().cpu().numpy()))
            print("loss: " + str(loss.detach().cpu().numpy()))
            y_pred = km.fit_predict(z.detach().cpu().numpy())
            cm = Clustering_Metrics(y, y_pred)
            acc, nmi, ari, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = cm.evaluationClusterModelFromLabel()
            pur = purity_score(y, y_pred)

            ##############################################################
            # Save logs  
            logdict = dict(iter=epoch, acc=acc, nmi=nmi, ari=ari, pur=pur, f1_macro=f1_macro, f1_micro=f1_micro, precision_macro=precision_macro, precision_micro=precision_micro, recall_macro=recall_macro, recall_micro=recall_micro, loss=loss.detach().cpu().numpy())
            logwriter.writerow(logdict)
            logfile.flush()

            if acc > acc_best:
                ##############################################################
                # Saving Graph Structures
                #graph = {'graph': adj_out_csr}
                #np.save(save_path + dataset + '/pretrain/graph_pretrain_'+ str(epoch) + '.npy', graph)

                ##############################################################
                # Saving 2D Embedded space
                #tsne = TSNE()
                #tsne_results = tsne.fit_transform(z.detach().cpu().numpy())
                #plt.figure()
                #sns.set(rc={'figure.figsize':(11.7,8.27)})
                #palette = sns.color_palette("bright", self.nClusters)
                #sns.set_style("white")
                #clusterviz = sns.scatterplot(tsne_results[:, 0], tsne_results[:, 1], hue=y, legend='brief', palette=palette)
                #plt.savefig(save_path + dataset + "/pretrain/vis_tsne_" + str(epoch) + ".png", dpi=400)
                
                ##############################################################
                # Initialize the centers
                centers = torch.tensor(km.cluster_centers_, dtype=torch.float, requires_grad=True) 
                self.assignment_1.state_dict()["cluster_centers"].copy_(centers)
                
                ##############################################################
                # Saving model
                acc_best = acc
                y_pred_best = y_pred
                torch.save(self.state_dict(), save_path + dataset + '/pretrain/model_pretrain.pk')
                #data = {'emb': z.detach().cpu().numpy(), 'hidden': hidden.detach().cpu().numpy()}
                #np.save(save_path + dataset + '/pretrain/data_pretrain.npy', data)

        print("Best accuracy : ", acc_best)  
        return y_pred_best, y
    
    def train(self, features, adj_norm, adj_label, adj, y, weight_tensor, norm, optimizer="Adam", epochs=200, lr=0.01, beta_1=0.3, beta_2=0.15, save_path="/home/mrabah_n/code/CVGAE/results/", dataset="Cora"):
        self.load_state_dict(torch.load(save_path + dataset + '/pretrain/model_pretrain.pk'))
        if optimizer == "Adam":
            opti = Adam(self.parameters(), lr=lr, weight_decay=1e-3)
        elif optimizer == "RMSProp":
            opti = RMSprop(self.parameters(), lr=lr, weight_decay=0.01)
        else:
            opti = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
        lr_s = StepLR(opti, step_size=120, gamma=0.9)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logfile = open(save_path + dataset + '/train/log_train.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'pur', 'f1_macro', 'f1_micro',
                                                        'precision_macro', 'precision_micro', 'recall_macro', 'recall_micro', 'acc_unconf', 'nmi_unconf',
                                                        'acc_conf', 'nmi_conf', 'nb_unconf', 'nb_conf', 'nb_links',
                                                        'nb_false_links', 'nb_true_links', 'nb_added_links', 'nb_false_added_links',
                                                        'nb_true_added_links', 'nb_dropped_links', 'nb_false_dropped_links',
                                                        'nb_true_dropped_links', 'Loss_recons', 'Loss_clus', 'Loss_reg',
                                                        'Loss_sep', 'Loss_comp', 'Loss'])
        logwriter.writeheader()
        epoch_bar = tqdm(range(epochs))

        ##############################################################
        # Preparing positive signals 
        adj_norm_pos = adj_norm
        adj_label_pos = adj_label
        features_pos = features
        norm_pos = norm

        ##############################################################
        # Training loop
        print("")
        print("Training......")
        acc_best = 0
        for epoch in epoch_bar:
            opti.zero_grad()
            z_mu_pos, z_sigma2_log_pos, emb_pos, hidden_pos = self.encode(features_pos, adj_norm_pos)
            p_pos = self.assignment_1(z_mu_pos)
            adj_out_pos = self.decode(emb_pos)

            if epoch % 10 == 0:
                unconflicted_ind, conflicted_ind = generate_unconflicted_data_index(p_pos, beta_1, beta_2)
                sep_ind = generate_sep_index(emb_pos, self.assignment_1.cluster_centers, p_pos)
                q_pos = target_distribution(p_pos, unconflicted_ind, conflicted_ind)
                adj_pos, adj_norm_pos, adj_label_pos, weight_tensor_pos, norm_pos, count_target_links = self.update_graph(adj, emb_pos, y, unconflicted_ind)
                count_links = evaluate_links(adj_pos, y)

            ##############################################################
            # Stopping condition
            if unconflicted_ind.shape[0] > (features_pos.shape[0] * 0.8):
                break

            ##############################################################
            # Loss
            Loss_recons = norm_pos * F.binary_cross_entropy(adj_out_pos.view(-1), adj_label_pos.to_dense().view(-1), weight=weight_tensor_pos)
            Loss_clus = 2 * self.kl_loss(torch.log(p_pos), q_pos)
            Loss_reg = torch.mean((1 / z_mu_pos.shape[0]) * torch.sum(z_mu_pos ** 2 + torch.exp(z_sigma2_log_pos) - 1 - z_sigma2_log_pos, dim=1))
            Loss_comp = Loss_recons + self.gamma_1 * Loss_clus + self.gamma_2 * Loss_reg
            Loss_sep = self.compute_separtion_loss(z_mu_pos, z_sigma2_log_pos, sep_ind, unconflicted_ind)
            Loss = Loss_comp + self.gamma_3 * Loss_sep

            ##############################################################
            # Evaluation
            acc_unconf, nmi_unconf, acc_conf, nmi_conf = self.compute_acc_and_nmi_conflicted_data(unconflicted_ind, conflicted_ind, emb_pos, y)
            epoch_bar.write('Loss training = {:.4f}'.format(Loss.detach().cpu().numpy()))
            print("")
            print("loss reconstruction: " + str(Loss_recons.detach().cpu().numpy()))
            print("loss clustering: " + str(Loss_clus.detach().cpu().numpy()))
            print("loss regularisation: " + str(Loss_reg.detach().cpu().numpy()))
            print("loss compactness: " + str(Loss_comp.detach().cpu().numpy()))
            print("loss separability: " + str(Loss_sep.detach().cpu().numpy()))
            print("loss: " + str(Loss.detach().cpu().numpy()))
            y_pred = self.predict(emb_pos)                            
            cm = Clustering_Metrics(y, y_pred)
            acc, nmi, ari, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = cm.evaluationClusterModelFromLabel()
            pur = purity_score(y, y_pred)

            ##############################################################
            # Update learnable parameters
            Loss.backward()
            opti.step()
            lr_s.step()

            ##############################################################
            # Save logs
            logdict = dict(iter=epoch, acc=acc, nmi=nmi, ari=ari, pur=pur, f1_macro=f1_macro, f1_micro=f1_micro,
                           precision_macro=precision_macro, precision_micro=precision_micro,
                           recall_macro=recall_macro, recall_micro=recall_micro,
                           acc_unconf=acc_unconf, nmi_unconf=nmi_unconf, acc_conf=acc_conf,
                           nb_links=count_links["nb_links"],
                           nb_false_links=count_links["nb_false_links"],
                           nb_true_links=count_links["nb_true_links"],
                           nb_added_links=count_target_links["nb_added_links"],
                           nb_false_added_links=count_target_links["nb_false_added_links"],
                           nb_true_added_links=count_target_links["nb_true_added_links"],
                           nb_dropped_links=count_target_links["nb_deleted_links"],
                           nb_false_dropped_links=count_target_links["nb_false_deleted_links"],
                           nb_true_dropped_links=count_target_links["nb_true_deleted_links"],
                           nmi_conf=nmi_conf, nb_unconf=unconflicted_ind.shape[0], nb_conf=conflicted_ind.shape[0],
                           Loss_recons=Loss_recons.detach().cpu().numpy(), Loss_clus=Loss_clus.detach().cpu().numpy(),
                           Loss_reg=Loss_reg.detach().cpu().numpy(), Loss_sep=Loss_sep.detach().cpu().numpy(),
                           Loss_comp=Loss_comp.detach().cpu().numpy(), Loss=Loss.detach().cpu().numpy())
            logwriter.writerow(logdict)
            logfile.flush()

            #if epoch % 50 == 0:
                ##############################################################
                # Saving Graph Structures
                #G = nx.convert_matrix.from_scipy_sparse_matrix(adj_pos)
                #i = 0
                #for node in G.nodes():
                #    G.nodes[node]['category'] = y[i]
                #    i += 1
                # put together a color map, one color for a category
                #color_map = {0:'b', 1:'g', 2:'r', 3:'c', 4:'m', 5:'y', 6:'k'}
                # construct a list of colors then pass to node_color
                #pos = nx.spring_layout(G)
                #nx.draw_networkx_nodes(G, pos , node_color=[color_map[G.nodes[node]['category']] for node in G.nodes()], alpha=0.6, node_size=20)
                #nx.draw_networkx_edges(G, pos, width=0.5, edge_color='0.75', alpha=0.5)
                #nx.write_graphml_lxml(G, save_path + dataset + "/train/adj_pos_epoch_" + str(epoch) + ".graphml")

                ##############################################################
                # TNSE
                #tsne = TSNE()
                #tsne_results = tsne.fit_transform(emb_pos.detach().cpu().numpy())
                #plt.figure()
                #sns.set(rc={'figure.figsize':(11.7,8.27)})
                #palette = sns.color_palette("bright", self.nClusters)
                #sns.set_style("white")
                #sns.scatterplot(tsne_results[:,0], tsne_results[:,1], hue=y, legend='brief', palette=palette)
                #plt.savefig(save_path + dataset + "/train/vis_tsne_" + str(epoch) + ".png", dpi=400)

            ##############################################################
            # Save model
            if (acc > acc_best):
                acc_best = acc
                y_pred_best = y_pred
                torch.save(self.state_dict(), save_path + dataset + '/train/model_cluster.pk')

        print("Best accuracy : ", acc_best)
        return y_pred_best, y

    def predict(self, z):
        p = self.assignment_1(z).detach().cpu().numpy()
        return np.argmax(p, axis=1)

    def encode(self, features, adj):
        hidden = self.base_gcn(features, adj)
        mean = self.gcn_mean(hidden, adj)
        log_sigma2 = self.gcn_logsigma2(hidden, adj)
        gaussian_noise = torch.randn(features.size(0), self.embedding_size).to("cuda:4")
        sampled_z = gaussian_noise * torch.exp(log_sigma2 / 2) + mean
        return mean, log_sigma2, sampled_z, hidden
            
    @staticmethod
    def decode(z):
        A_pred = torch.sigmoid(torch.matmul(z, z.t()))
        return A_pred

    def compute_acc_and_nmi_conflicted_data(self, unconf_indices, conf_indices, emb, y):
        if unconf_indices.size == 0:
            print(' '*8 + "Empty list of unconflicted data")
            acc_unconf = 0
            nmi_unconf = 0
        else:
            print("\nNumber of Unconflicted points : ", len(unconf_indices))
            emb_unconf = emb[unconf_indices]
            y_unconf = y[unconf_indices]
            y_pred_unconf = self.predict(emb_unconf)
            acc_unconf = mt.acc(y_unconf, y_pred_unconf)
            nmi_unconf = mt.nmi(y_unconf, y_pred_unconf)
            print(' '*8 + '|==>  acc unconflicted data: %.4f,  nmi unconflicted data: %.4f  <==|'% (acc_unconf, nmi_unconf))

        if conf_indices.size == 0:
            print(' '*8 + "Empty list of conflicted data")
            acc_conf = 0
            nmi_conf = 0
        else:
            print("Number of conflicted points : ", len(conf_indices))
            emb_conf = emb[conf_indices] 
            y_conf = y[conf_indices]
            y_pred_conf = self.predict(emb_conf)
            acc_conf = mt.acc(y_conf, y_pred_conf)
            nmi_conf = mt.nmi(y_conf, y_pred_conf)
            print(' '*8 + '|==>  acc conflicted data: %.4f,  nmi conflicted data: %.4f  <==|'% (mt.acc(y_conf, y_pred_conf), mt.nmi(y_conf, y_pred_conf)))    
        return acc_unconf, nmi_unconf, acc_conf, nmi_conf

                 

  

