# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 20:51:38 2018
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise, adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import pickle
import os.path
from data_preprocess_t1 import data_preparation_type_I, get_validation_set_features
from data_preprocess_t2 import data_process_type_II
from mpl_toolkits.mplot3d import axes3d # this is needed for projection=3d
import warnings
warnings.filterwarnings("ignore")

NUM_CLUSTER = 4


def visualize_matrix(M, figsize, title):
    """
    Plot a matrix as heatmap
    """
    plt.figure(figsize=figsize)
    plt.imshow(M)
    plt.colorbar()
    plt.xticks(range(M.shape[1]), M.columns.tolist(), rotation='vertical')
    plt.yticks(range(M.shape[0]), M.index.tolist())
    plt.tight_layout()
    plt.savefig(title+'.png')


def graph_visualize(df1, df2=None):
    """
    Use networkx to visualize the correlation between two dataframes.
    
    Each node is one column.
    Each edge represents the correlation between two columns.
    """
    df1 = df1.fillna(0)
    
    G = nx.Graph()
    nodes = df1.columns.tolist()
    G.add_nodes_from(nodes)
    
    n = len(nodes)
    W = pairwise.cosine_similarity(df1.values.T)
    for i in range(n):
        for j in range(i+1,n):
            if W[i][j] > 0.8:
                G.add_edge(nodes[i], nodes[j], weight=W[i][j])
    
    plt.figure(figsize=(16,16))
    nx.draw(G, with_labels=True)
    return W, G


def kmeans_clustering(df, title, n_cluster=NUM_CLUSTER, isplot=False):
    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    km_res = kmeans.fit(df)
    if isplot:
        centers = pd.DataFrame(km_res.cluster_centers_,
                               index=['Cluster {0}'.format(e) for e in range(1, n_cluster+1)],
                               columns=df.columns.tolist())
        figheight = 4 if df.shape[1]/2 < 4 else df.shape[1]/2
        visualize_matrix(centers, (figheight, 4), title)
    return km_res


def save_clustering_centers(df, labels, title, n_cluster=NUM_CLUSTER):
    data_columns = df.columns.tolist()
    df['labels'] = labels
    centers = []
    for i in range(n_cluster):
        centers.append(df.loc[df['labels'] == i, data_columns].mean().values)
    centers = pd.DataFrame(centers, columns=data_columns,
                           index=['Cluster {0}'.format(e) for e in range(1, n_cluster+1)])
    centers.to_csv(title+".csv")


def worker(para):
    """
    Iterative feature selection parallel workers
    """
    feature, fcm, df, kmeans, labels_org = para
    columns = fcm[feature]
    df_p = df.drop(columns, axis=1)
    labels_p = kmeans.fit(df_p).labels_
    ri = adjusted_rand_score(labels_p, labels_org)
    # print "Test feature {0}\t r = {1}".format(column, ri)
    return feature, ri


def select_features_to_drop(df, fcm, kmeans, labels_org):
    """
    Drop one feature from df according to clustering random index.
    
    Iterate over every feature f:
        df - f = df_p
        Compute  the clustering(df) v.s. clustering(df_p)
    
    Drop f with the highest rand_score.
    """
    pool = Pool(processes=cpu_count())
    
    n = len(fcm)  # feature_columns_map
    ris = list(
        pool.imap_unordered(
            worker,
            zip(fcm.keys(), [fcm]*n, [df]*n, [kmeans]*n, [labels_org]*n),
            chunksize=4))

    ris = sorted(ris, key=lambda x: -x[1])
    
    pool.close()
    pool.join()
    column_to_drop = fcm[ris[0][0]]
    print("Feature {0} is dropped with ri = {1}".format(*ris[0]))
    fcm.pop(ris[0][0])  # drop feature
    return df.drop(column_to_drop, axis=1), ris[0][1], fcm


def cluster_correlations_by_frequency(cls1, cls2):
    labelsA = cls1.labels_
    labelsB = cls2.labels_
    C = np.zeros((NUM_CLUSTER, NUM_CLUSTER))
    for i in range(NUM_CLUSTER):
        idxA = np.where(labelsA == i)[0]
        for j in range(NUM_CLUSTER):
            idxB = np.where(labelsB == j)[0]
            C[i][j] = len(np.intersect1d(idxA, idxB))
    c_df = pd.DataFrame(C,
                        index=['cls-{0} from T1'.format(e) for e in range(NUM_CLUSTER)],
                        columns=['cls-{0} from T2'.format(e) for e in range(NUM_CLUSTER)])
    visualize_matrix(c_df, (6,6), 'frequency-similarity-two-level')
    return C


def iterative_feature_selection(df, feature_columns_map,
                                columns_to_drop=[],
                                pickle_to="remained_columns_type1.pickle"):
    """
    Use random index to iteratively remove least important features
    """
    ri = 1
    idx = 0
    
    df_dropped = df.drop(columns_to_drop, axis=1)
    # original labels
    kmeans = KMeans(n_clusters=NUM_CLUSTER, random_state=0)
    labels_org = kmeans.fit(df).labels_
    while ri > 0.8:
        print "Iteration {0}".format(idx+1)
        df_prv = df_dropped
        df_dropped, ri, feature_columns_map = \
            select_features_to_drop(df_prv, feature_columns_map, kmeans, labels_org)
        idx += 1
    
    pickle.dump(df_prv.columns.tolist(), open(pickle_to, 'w'))


def visualize_clusters_3d(df, labels, figpath):
    print "start PCA ..."
    xy = PCA(n_components=3).fit_transform(df)
    xy_df = pd.DataFrame(xy, columns=['x', 'y', 'z'])
    xy_df['labels'] = labels
    f = plt.figure(figsize=(8, 8))
    ax = f.add_subplot(111, projection='3d')
    colors = ['red', 'blue', 'green', 'black', 'yellow', 'cyan', 'magenta']
    legend_labels = []
    for i in range(NUM_CLUSTER):
        rows_selected = xy_df['labels'] == i
        xy_selected = xy_df[rows_selected]
        print "plot clusters {0} with {1} samples".format(i+1, len(xy_selected))
        if len(xy_selected) > 10000:
            xy_selected = xy_selected.sample(10000)
        ax.scatter(xy_selected['x'], xy_selected['y'], xy_selected['z'], c=colors[i])
        legend_labels.append("Cluster {}".format(i))
    ax.set_xlabel("PCA componet 1")
    ax.set_ylabel("PCA componet 2")
    ax.set_zlabel("PCA componet 3")
    plt.legend(legend_labels)
    f.tight_layout()
    f.show()
    f.savefig(figpath)
    print "PCA visualization finished"


def visualize_clusters_2d(df, labels, figpath, n_cluster=NUM_CLUSTER):
    use_pca = df.shape[1] > 2
    if use_pca:
        xy = PCA(n_components=2).fit_transform(df)
        xy_df = pd.DataFrame(xy, columns=['x', 'y'])        
    else:
        xy_df = pd.DataFrame(df.values, columns=['x', 'y'])
    xy_df['labels'] = labels        
    f = plt.figure(figsize=(8, 8))
    ax = f.add_subplot(111)
    colors = ['red', 'blue', 'green', 'black', 'yellow', 'cyan', 'magenta']
    legend_labels = []
    for i in range(n_cluster):
        rows_selected = xy_df['labels'] == i
        xy_selected = xy_df[rows_selected]
        if len(xy_selected) > 10000:
            xy_selected = xy_selected.sample(10000)
        ax.scatter(xy_selected['x'], xy_selected['y'], c=colors[i])
        legend_labels.append("Cluster {}".format(i))
    if use_pca:
        ax.set_xlabel("PCA componet 1")
        ax.set_ylabel("PCA componet 2")
    else:
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])
    plt.legend(legend_labels)
    plt.show(block=False)
    plt.tight_layout()
    f.savefig(figpath)


def tune_optimal_cluster_numbers(t1_features):
    pickle_fname = "tune_cluster_cnt.pickle"
    if os.path.isfile(pickle_fname):
        with open(pickle_fname, 'r') as fin:
            x = pickle.load(fin)
            silhouettes = pickle.load(fin)
            SSE = pickle.load(fin)
    else:
        x = range(2, 21)
        silhouettes = []
        SSE = []
        for i in x:
            c = kmeans_clustering(t1_features, 'tmp', i)
            silhouette = silhouette_score(t1_features.values, c.labels_, sample_size=20000, random_state=2018)
            print i, silhouette, c.inertia_
            silhouettes.append(silhouette)
            SSE.append(c.inertia_)
        with open(pickle_fname, 'w') as fout:
            pickle.dump(x, fout)
            pickle.dump(silhouettes, fout)
            pickle.dump(SSE, fout)
    # plot
    f, ax1 = plt.subplots()
    ax1.plot(x, silhouettes, lw=2, ls='-', c='blue', label='Silhouette')
    ax1.axvline(x=4, ls=':', lw=2, c='green')
    ax1.set_xlabel("Number of clusters", fontsize=16)
    ax1.set_ylabel("Silhouette scores", fontsize=16)
    ax2 = ax1.twinx()
    ax2.plot(x, SSE, lw=2, ls='-.', c='red', label='SSE')
    ax2.set_ylabel("SSE", fontsize=16)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0, fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    f.tight_layout()
    plt.show()
    f.savefig("tune_cluster_cnt.jpeg")


"""
Some one time utility scripts
"""


def process_iterative_feature_selection_log():
    """
    Process iterative feature selection log into <feature, rand_index> pair.
    :return:
    """
    import re
    regstr = r"Feature (.+) is dropped with ri = (.+)"
    with open("iterative_feature_selection.log") as fin:
        text = fin.read()
    matches = re.finditer(regstr, text, re.MULTILINE)

    with open("iterative_feature_selection.txt", "w") as fout:
        for match in matches:
            fout.write("{} {}\n".format(*match.groups()))


def select_type1_by_rand_index(ri_threshold):
    fname = "type1_remained_df.pickle"
    if not os.path.isfile(fname):
        _, t1_features, fcm, _, _ = data_preparation_type_I()
        columns_to_drop = []
        with open("iterative_feature_selection.txt") as fin:
            for line in fin:
                ls = line.strip().split(" ")
                feature = ls[0]
                ri = float(ls[1])
                if ri > ri_threshold:
                    columns_to_drop += fcm[feature]
                    fcm.pop(feature)
                    print feature, ri
        t1_features = t1_features.drop(columns_to_drop, axis=1)
        with open(fname, "w") as fout:
            pickle.dump(t1_features, fout)
            pickle.dump(fcm, fout)
    else:
        with open(fname, "r") as fin:
            t1_features = pickle.load(fin)
            fcm = pickle.load(fin)
    assert t1_features.shape[1] == sum([len(v) for v in fcm.values()])
    return t1_features, fcm


if __name__ == '__main__':
    options = 'visualize_clusters'
    if options == 'tune_cluster_numbers':
        t1_df, t1_features, feature_columns_map, validation, df = data_preparation_type_I()
        tune_optimal_cluster_numbers(t1_features)
    elif options == 'feature_selection':
        t1_df, t1_features, feature_columns_map, validation, df = data_preparation_type_I()
        # Caution: this function runs over 1 hour on 4 threads
        iterative_feature_selection(t1_features, feature_columns_map, pickle_to="type1_remained_columns.pickle")
    elif options == "get_remained_type1_df":
        t1_features, fcm = select_type1_by_rand_index(ri_threshold=0.95)
    elif options == 'visualize_clusters':
        _, _, _, _, df = data_preparation_type_I()
        t1_features, fcm = select_type1_by_rand_index(ri_threshold=0.9)
        print "{} questions remain".format(len(fcm))
        cls1 = kmeans_clustering(t1_features, 'type1-kmeans', isplot=True)
        save_clustering_centers(t1_features, cls1.labels_, "type1-kmeans-centers-selected")
        visualize_clusters_3d(t1_features, cls1.labels_, "type1_pca_3d_selected.png")

        data_process_type_II(df, cls1)

        # w, g = graph_visualize(s)
        # visualize_matrix(pd.DataFrame(w, index=s.columns.tolist(), columns=s.columns.tolist()),
        #                  (16, 16), 'pairwise-feature-similarity')

        # C = cluster_correlations_by_frequency(cls1, cls2)




