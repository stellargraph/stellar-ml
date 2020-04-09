# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Mappers to provide input data for the graph models in layers.

"""
__all__ = ["ClusterNodeGenerator", "ClusterNodeSequence"]

import random
import copy
import numpy as np
import networkx as nx
from tensorflow.keras.utils import Sequence
import tensorflow as tf

from scipy import sparse
from ..core.graph import StellarGraph
from ..core.utils import is_real_iterable, normalize_adj


class ClusterNodeGenerator:
    """
    A data generator for use with ClusterGCN models on homogeneous graphs, [1].

    The supplied graph G should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features to be available for all
    nodes in the graph.
    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    This generator will supply the features array and the adjacency matrix to a
    mini-batch Keras graph ML model.

    [1] `W. Chiang, X. Liu, S. Si, Y. Li, S. Bengio, C. Hsieh, 2019 <https://arxiv.org/abs/1905.07953>`_.

    For more information, please see the ClusterGCN demo: `<https://github.com/stellargraph/stellargraph/blob/master/demos/>`_

    Args:
        G (StellarGraph): a machine-learning StellarGraph-type graph
        clusters (int or list): If int then it indicates the number of clusters (default is 1 that is the given graph).
            If clusters is greater than 1, then nodes are uniformly at random assigned to a cluster. If list,
            then it should be a list of lists of node IDs such that each list corresponds to a cluster of nodes
            in G. The clusters should be non-overlapping.
        q (float): The number of clusters to combine for each mini-batch. The default is 1.
        lam (float): The mixture coefficient for adjacency matrix normalisation.
        name (str): an optional name of the generator
    """

    def __init__(self, G, clusters=1, q=1, lam=0.1, name=None):

        if not isinstance(G, StellarGraph):
            raise TypeError("Graph must be a StellarGraph or StellarDiGraph object.")

        self.graph = G
        self.name = name
        self.q = q  # The number of clusters to sample per mini-batch
        self.lam = lam
        self.clusters = clusters

        if isinstance(clusters, list):
            self.k = len(clusters)
        elif isinstance(clusters, int):
            if clusters <= 0:
                raise ValueError(
                    "{}: clusters must be greater than 0.".format(type(self).__name__)
                )
            self.k = clusters
        else:
            raise TypeError(
                "{}: clusters must be either int or list type.".format(
                    type(self).__name__
                )
            )

        # Some error checking on the given parameter values
        if not isinstance(lam, float):
            raise TypeError("{}: lam must be a float type.".format(type(self).__name__))

        if lam < 0 or lam > 1:
            raise ValueError(
                "{}: lam must be in the range [0, 1].".format(type(self).__name__)
            )

        if not isinstance(q, int):
            raise TypeError("{}: q must be integer type.".format(type(self).__name__))

        if q <= 0:
            raise ValueError(
                "{}: q must be greater than 0.".format(type(self).__name__)
            )

        if self.k % q != 0:
            raise ValueError(
                "{}: the number of clusters must be exactly divisible by q.".format(
                    type(self).__name__
                )
            )

        # Check if the graph has features
        G.check_graph_for_ml()

        self.node_list = list(G.nodes())

        # Check that there is only a single node type
        if len(G.node_types) > 1:
            raise ValueError(
                "{}: node generator requires graph with single node type; "
                "a graph with multiple node types is passed. Stopping.".format(
                    type(self).__name__
                )
            )

        self.feature_size = next(iter(G.node_feature_sizes().values()))

        if isinstance(clusters, int):
            # We are not given graph clusters.
            # We are going to split the graph into self.k random clusters
            all_nodes = list(G.nodes())
            random.shuffle(all_nodes)
            cluster_size = len(all_nodes) // self.k
            self.clusters = [
                all_nodes[i : i + cluster_size]
                for i in range(0, len(all_nodes), cluster_size)
            ]
            if len(self.clusters) > self.k:
                # for the case that the number of nodes is not exactly divisible by k, we combine
                # the last cluster with the second last one
                self.clusters[-2].extend(self.clusters[-1])
                del self.clusters[-1]

        print(f"Number of clusters {self.k}")
        for i, c in enumerate(self.clusters):
            print(f"{i} cluster has size {len(c)}")

    def flow(self, node_ids, targets=None, name=None):
        """
        Creates a generator/sequence object for training, evaluation, or prediction
        with the supplied node ids and numeric targets.

        Args:
            node_ids (iterable): an iterable of node ids for the nodes of interest
                (e.g., training, validation, or test set nodes)
            targets (2d array, optional): a 2D array of numeric node targets with shape `(len(node_ids),
                target_size)`
            name (str, optional): An optional name for the returned generator object.

        Returns:
            A ClusterNodeSequence object to use with ClusterGCN in Keras
            methods :meth:`fit`, :meth:`evaluate`, and :meth:`predict`

        """
        if targets is not None:
            # Check targets is an iterable
            if not is_real_iterable(targets):
                raise TypeError(
                    "{}: Targets must be an iterable or None".format(
                        type(self).__name__
                    )
                )

            # Check targets correct shape
            if len(targets) != len(node_ids):
                raise ValueError(
                    "{}: Targets must be the same length as node_ids".format(
                        type(self).__name__
                    )
                )

        return ClusterNodeSequence(
            self.graph,
            self.clusters,
            targets=targets,
            node_ids=node_ids,
            q=self.q,
            lam=self.lam,
            name=name,
        )


class ClusterNodeSequence(Sequence):
    """
    A Keras-compatible data generator for node inference using ClusterGCN model.
    Use this class with the Keras methods :meth:`keras.Model.fit`,
        :meth:`keras.Model.evaluate`, and
        :meth:`keras.Model.predict`,

    This class should be created using the `.flow(...)` method of
    :class:`ClusterNodeGenerator`.

    Args:
        graph (StellarGraph): The graph
        clusters (list): A list of lists such that each sub-list indicates the nodes in a cluster.
            The length of this list, len(clusters) indicates the number of batches in one epoch.
        targets (np.ndarray, optional): An optional array of node targets of size (N x C),
            where C is the target size (e.g., number of classes for one-hot class targets)
        node_ids (iterable, optional): The node IDs for the target nodes. Required if targets is not None.
        normalize_adj (bool, optional): Specifies whether the adjacency matrix for each mini-batch should
            be normalized or not. The default is True.
        q (int, optional): The number of subgraphs to combine for each batch. The default value is
            1 such that the generator treats each subgraph as a batch.
        lam (float, optional): The mixture coefficient for adjacency matrix normalisation (the
            'diagonal enhancement' method). Valid values are in the interval [0, 1] and the default value is 0.1.
        name (str, optional): An optional name for this generator object.
    """

    def __init__(
        self,
        graph,
        clusters,
        targets=None,
        node_ids=None,
        normalize_adj=True,
        q=1,
        lam=0.1,
        name=None,
    ):

        self.name = name
        self.clusters = list()
        self.clusters_original = copy.deepcopy(clusters)
        self.graph = graph
        self.node_list = list(graph.nodes())
        self.normalize_adj = normalize_adj
        self.q = q
        self.lam = lam
        self.node_order = list()
        self._node_order_in_progress = list()
        self.__node_buffer = dict()
        self.target_ids = list()

        if len(clusters) % self.q != 0:
            raise ValueError(
                "The number of clusters should be exactly divisible by q. However, {} number of clusters is not exactly divisible by {}.".format(
                    len(clusters), q
                )
            )

        if node_ids is not None:
            self.target_ids = list(node_ids)

        if targets is not None:
            if node_ids is None:
                raise ValueError(
                    "Since targets is not None, node_ids must be given and cannot be None."
                )

            if len(node_ids) != len(targets):
                raise ValueError(
                    "When passed together targets and indices should be the same length."
                )

            self.targets = np.asanyarray(targets)
            self.target_node_lookup = dict(
                zip(self.target_ids, range(len(self.target_ids)))
            )
        else:
            self.targets = None

        self.on_epoch_end()

    def __len__(self):
        num_batches = len(self.clusters_original) // self.q
        return num_batches

    def _diagonal_enhanced_normalization(self, adj_cluster):
        # Cluster-GCN normalization is:
        #     A~ + λdiag(A~) where A~ = N(A + I) with normalization factor N = (D + I)^(-1)
        #
        # Expands to:
        #     NA + NI + λN(diag(A) + I) =
        #     NA + N(I + λ(diag(A) + I)) =
        #     NA + λN(diag(A) + (1 + 1/λ)I))
        #
        # (This could potentially become a layer, to benefit from a GPU.)
        degrees = np.asarray(adj_cluster.sum(axis=1)).ravel()
        normalization = 1 / (degrees + 1)

        # NA: multiply rows manually
        norm_adj = adj_cluster.multiply(normalization[:, None]).toarray()

        # λN(diag(A) + (1 + 1/λ)I): work with the diagonals directly
        diag = np.diag(norm_adj)
        diag_addition = (
            normalization * self.lam * (adj_cluster.diagonal() + (1 + 1 / self.lam))
        )
        np.fill_diagonal(norm_adj, diag + diag_addition)
        return norm_adj

    def __getitem__(self, index):
        # The next batch should be the adjacency matrix for the cluster and the corresponding feature vectors
        # and targets if available.
        cluster = self.clusters[index]
        adj_cluster = self.graph.to_adjacency_matrix(cluster)

        if self.normalize_adj:
            adj_cluster = self._diagonal_enhanced_normalization(adj_cluster)
        else:
            adj_cluster = adj_cluster.toarray()

        g_node_list = list(cluster)

        # Determine the target nodes that exist in this cluster
        target_nodes_in_cluster = np.asanyarray(
            list(set(g_node_list).intersection(self.target_ids))
        )

        self.__node_buffer[index] = target_nodes_in_cluster

        # Dictionary to store node indices for quicker node index lookups
        node_lookup = dict(zip(g_node_list, range(len(g_node_list))))

        # The list of indices of the target nodes in self.node_list
        target_node_indices = np.array(
            [node_lookup[n] for n in target_nodes_in_cluster]
        )

        if index == (len(self.clusters_original) // self.q) - 1:
            # last batch
            self.__node_buffer_dict_to_list()

        cluster_targets = None
        #
        if self.targets is not None:
            # Dictionary to store node indices for quicker node index lookups
            # The list of indices of the target nodes in self.node_list
            cluster_target_indices = np.array(
                [self.target_node_lookup[n] for n in target_nodes_in_cluster]
            )
            cluster_targets = self.targets[cluster_target_indices]
            cluster_targets = cluster_targets.reshape((1,) + cluster_targets.shape)

        with tf.device("/CPU:0"):
            features = self.graph.node_features_tensors(g_node_list)
            features = tf.reshape(features, (1,) + features.shape)

        adj_cluster = adj_cluster.reshape((1,) + adj_cluster.shape)
        target_node_indices = target_node_indices[np.newaxis, :]

        return [features, target_node_indices, adj_cluster], cluster_targets

    def __node_buffer_dict_to_list(self):
        self.node_order = []
        for k, v in self.__node_buffer.items():
            self.node_order.extend(v)

    def on_epoch_end(self):
        """
         Shuffle all nodes at the end of each epoch
        """
        if self.q > 1:
            # combine clusters
            cluster_indices = list(range(len(self.clusters_original)))
            random.shuffle(cluster_indices)
            self.clusters = []

            for i in range(0, len(cluster_indices) - 1, self.q):
                cc = cluster_indices[i : i + self.q]
                tmp = []
                for l in cc:
                    tmp.extend(list(self.clusters_original[l]))
                self.clusters.append(tmp)
        else:
            self.clusters = copy.deepcopy(self.clusters_original)

        self.__node_buffer = dict()

        random.shuffle(self.clusters)
