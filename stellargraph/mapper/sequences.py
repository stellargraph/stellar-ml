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
Sequences to provide input to Keras

"""
__all__ = [
    "NodeSequence",
    "LinkSequence",
    "OnDemandLinkSequence",
    "FullBatchSequence",
    "SparseFullBatchSequence",
    "RelationalFullBatchNodeSequence",
    "GraphSequence",
    "CorruptedNodeSequence",
]

import warnings
import operator
import random
import collections
import numpy as np
import itertools as it
import networkx as nx
import scipy.sparse as sps
from tensorflow.keras import backend as K
from functools import reduce
from tensorflow.keras.utils import Sequence
from ..data.unsupervised_sampler import UnsupervisedSampler
from ..core.utils import is_real_iterable, normalize_adj
from ..random import random_state
from scipy import sparse
from ..core.experimental import experimental
import tensorflow as tf


class NodeSequence(Sequence):
    """Keras-compatible data generator to use with the Keras
    methods :meth:`keras.Model.fit`, :meth:`keras.Model.evaluate`,
    and :meth:`keras.Model.predict`.

    This class generated data samples for node inference models
    and should be created using the `.flow(...)` method of
    :class:`GraphSAGENodeGenerator` or :class:`DirectedGraphSAGENodeGenerator`
    or :class:`HinSAGENodeGenerator` or :class:`Attri2VecNodeGenerator`.

    These generator classes are used within the NodeSequence to generate
    the required features for downstream ML tasks from the graph.

    Args:
        sample_function (Callable): A function that returns features for supplied head nodes.
        ids (list): A list of the node_ids to be used as head-nodes in the downstream task.
        targets (list, optional): A list of targets or labels to be used in the downstream task.
        shuffle (bool): If True (default) the ids will be randomly shuffled every epoch.
    """

    def __init__(
        self, sample_function, batch_size, ids, targets=None, shuffle=True, seed=None
    ):
        # Check that ids is an iterable
        if not is_real_iterable(ids):
            raise TypeError("IDs must be an iterable or numpy array of graph node IDs")

        # Check targets is iterable & has the correct length
        if targets is not None:
            if not is_real_iterable(targets):
                raise TypeError("Targets must be None or an iterable or numpy array ")
            if len(ids) != len(targets):
                raise ValueError(
                    "The length of the targets must be the same as the length of the ids"
                )
            self.targets = np.asanyarray(targets)
        else:
            self.targets = None

        # Store the generator to draw samples from graph
        if isinstance(sample_function, collections.abc.Callable):
            self._sample_function = sample_function
        else:
            raise TypeError(
                "({}) The sampling function expects a callable function.".format(
                    type(self).__name__
                )
            )

        self.ids = list(ids)
        self.data_size = len(self.ids)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self._rs, _ = random_state(seed)

        # Shuffle IDs to start
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.data_size / self.batch_size))

    def __getitem__(self, batch_num):
        """
        Generate one batch of data

        Args:
            batch_num (int): number of a batch

        Returns:
            batch_feats (list): Node features for nodes and neighbours sampled from a
                batch of the supplied IDs
            batch_targets (list): Targets/labels for the batch.

        """
        start_idx = self.batch_size * batch_num
        end_idx = start_idx + self.batch_size
        if start_idx >= self.data_size:
            raise IndexError("Mapper: batch_num larger than length of data")
        # print("Fetching batch {} [{}]".format(batch_num, start_idx))

        # The ID indices for this batch
        batch_indices = self.indices[start_idx:end_idx]

        # Get head (root) nodes
        head_ids = [self.ids[ii] for ii in batch_indices]

        # Get corresponding targets
        batch_targets = None if self.targets is None else self.targets[batch_indices]

        # Get features for nodes
        batch_feats = self._sample_function(head_ids, batch_num)

        return batch_feats, batch_targets

    def on_epoch_end(self):
        """
        Shuffle all head (root) nodes at the end of each epoch
        """
        self.indices = list(range(self.data_size))
        if self.shuffle:
            self._rs.shuffle(self.indices)


class LinkSequence(Sequence):
    """
    Keras-compatible data generator to use with Keras methods :meth:`keras.Model.fit`,
    :meth:`keras.Model.evaluate`, and :meth:`keras.Model.predict`
    This class generates data samples for link inference models
    and should be created using the :meth:`flow` method of
    :class:`GraphSAGELinkGenerator` or :class:`HinSAGELinkGenerator` or :class:`Attri2VecLinkGenerator`.

    Args:
        sample_function (Callable): A function that returns features for supplied head nodes.
        ids (iterable): Link IDs to batch, each link id being a tuple of (src, dst) node ids.
        targets (list, optional): A list of targets or labels to be used in the downstream task.
        shuffle (bool): If True (default) the ids will be randomly shuffled every epoch.
        seed (int, optional): Random seed
    """

    def __init__(
        self, sample_function, batch_size, ids, targets=None, shuffle=True, seed=None
    ):
        # Check that ids is an iterable
        if not is_real_iterable(ids):
            raise TypeError("IDs must be an iterable or numpy array of graph node IDs")

        # Check targets is iterable & has the correct length
        if targets is not None:
            if not is_real_iterable(targets):
                raise TypeError("Targets must be None or an iterable or numpy array ")
            if len(ids) != len(targets):
                raise ValueError(
                    "The length of the targets must be the same as the length of the ids"
                )
            self.targets = np.asanyarray(targets)
        else:
            self.targets = None

        # Ensure number of labels matches number of ids
        if targets is not None and len(ids) != len(targets):
            raise ValueError("Length of link ids must match length of link targets")

        # Store the generator to draw samples from graph
        if isinstance(sample_function, collections.abc.Callable):
            self._sample_features = sample_function
        else:
            raise TypeError(
                "({}) The sampling function expects a callable function.".format(
                    type(self).__name__
                )
            )

        self.batch_size = batch_size
        self.ids = list(ids)
        self.data_size = len(self.ids)
        self.shuffle = shuffle
        self._rs, _ = random_state(seed)

        # Shuffle the IDs to begin
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.data_size / self.batch_size))

    def __getitem__(self, batch_num):
        """
        Generate one batch of data
        Args:
            batch_num (int): number of a batch
        Returns:
            batch_feats (list): Node features for nodes and neighbours sampled from a
                batch of the supplied IDs
            batch_targets (list): Targets/labels for the batch.
        """
        start_idx = self.batch_size * batch_num
        end_idx = start_idx + self.batch_size

        if start_idx >= self.data_size:
            raise IndexError("Mapper: batch_num larger than length of data")
        # print("Fetching {} batch {} [{}]".format(self.name, batch_num, start_idx))

        # The ID indices for this batch
        batch_indices = self.indices[start_idx:end_idx]

        # Get head (root) nodes for links
        head_ids = [self.ids[ii] for ii in batch_indices]

        # Get targets for nodes
        batch_targets = None if self.targets is None else self.targets[batch_indices]

        # Get node features for batch of link ids
        batch_feats = self._sample_features(head_ids, batch_num)

        return batch_feats, batch_targets

    def on_epoch_end(self):
        """
        Shuffle all link IDs at the end of each epoch
        """
        self.indices = list(range(self.data_size))
        if self.shuffle:
            self._rs.shuffle(self.indices)


class OnDemandLinkSequence(Sequence):
    """
    Keras-compatible data generator to use with Keras methods :meth:`keras.Model.fit`,
    :meth:`keras.Model.evaluate`, and :meth:`keras.Model.predict`

    This class generates data samples for link inference models
    and should be created using the :meth:`flow` method of
    :class:`GraphSAGELinkGenerator` or :class:`Attri2VecLinkGenerator`.

    Args:
        sample_function (Callable): A function that returns features for supplied head nodes.
        sampler (UnsupersizedSampler):  An object that encapsulates the neighbourhood sampling of a graph.
            The generator method of this class returns a batch of positive and negative samples on demand.
    """

    def __init__(self, sample_function, batch_size, walker, shuffle=True):
        # Store the generator to draw samples from graph
        if isinstance(sample_function, collections.abc.Callable):
            self._sample_features = sample_function
        else:
            raise TypeError(
                "({}) The sampling function expects a callable function.".format(
                    type(self).__name__
                )
            )

        if not isinstance(walker, UnsupervisedSampler):
            raise TypeError(
                "({}) UnsupervisedSampler is required.".format(type(self).__name__)
            )

        self.batch_size = batch_size
        self.walker = walker
        self.shuffle = shuffle
        # FIXME(#681): all batches are created at once, so this is no longer "on demand"
        self._batches = self._create_batches()
        self.length = len(self._batches)
        self.data_size = sum(len(batch[0]) for batch in self._batches)

    def __getitem__(self, batch_num):
        """
        Generate one batch of data.

        Args:
            batch_num<int>: number of a batch

        Returns:
            batch_feats<list>: Node features for nodes and neighbours sampled from a
                batch of the supplied IDs
            batch_targets<list>: Targets/labels for the batch.

        """

        if batch_num >= self.__len__():
            raise IndexError(
                "Mapper: batch_num larger than number of esstaimted  batches for this epoch."
            )
        # print("Fetching {} batch {} [{}]".format(self.name, batch_num, start_idx))

        # Get head nodes and labels
        head_ids, batch_targets = self._batches[batch_num]

        # Obtain features for head ids
        batch_feats = self._sample_features(head_ids, batch_num)

        return batch_feats, batch_targets

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.length

    def _create_batches(self):
        return self.walker.run(self.batch_size)

    def on_epoch_end(self):
        """
        Shuffle all link IDs at the end of each epoch
        """
        if self.shuffle:
            self._batches = self._create_batches()


def _full_batch_tensor_and_reshape(x, propagate_none=False):
    """
    Args:
        array: an array-like object
        propagate_none: if True, return None when array is None
    Returns:
        array as a numpy array with an extra first dimension (batch dimension) equal to 1
    """
    # if it's ok, just short-circuit on None (e.g. for target arrays, that may or may not exist)
    if propagate_none and x is None:
        return None

    with tf.device("/CPU:0"):
        if not isinstance(x, tf.Tensor):
            x = tf.constant(x)

        x = tf.expand_dims(x, axis=0)
    return x


class FullBatchSequence(Sequence):
    """
    Keras-compatible data generator for for node inference models
    that require full-batch training (e.g., GCN, GAT).
    Use this class with the Keras methods :meth:`keras.Model.fit`,
        :meth:`keras.Model.evaluate`, and
        :meth:`keras.Model.predict`,

    This class should be created using the `.flow(...)` method of
    :class:`FullBatchNodeGenerator`.

    Args:
        features (np.ndarray): An array of node features of size (N x F),
            where N is the number of nodes in the graph, F is the node feature size
        A (np.ndarray or sparse matrix): An adjacency matrix of the graph of size (N x N).
        targets (np.ndarray, optional): An optional array of node targets of size (N x C),
            where C is the target size (e.g., number of classes for one-hot class targets)
        indices (np.ndarray, optional): Array of indices to the feature and adjacency matrix
            of the targets. Required if targets is not None.
    """

    use_sparse = False

    def __init__(self, features, A, targets=None, indices=None):

        if (targets is not None) and (len(indices) != len(targets)):
            raise ValueError(
                "When passed together targets and indices should be the same length."
            )

        # Store features as is and targets as tensors
        self.features = _full_batch_tensor_and_reshape(features)
        self.target_indices = _full_batch_tensor_and_reshape(indices)

        # Convert sparse matrix to dense:
        if sps.issparse(A) and hasattr(A, "toarray"):
            self.A_dense = _full_batch_tensor_and_reshape(A.toarray())
        elif isinstance(A, (np.ndarray, np.matrix)):
            self.A_dense = _full_batch_tensor_and_reshape(A)
        else:
            raise TypeError(
                "Expected input matrix to be either a Scipy sparse matrix or a Numpy array."
            )

        # Reshape all inputs to have batch dimension of 1
        self.target_indices = _full_batch_tensor_and_reshape(indices)
        self.inputs = [self.features, self.target_indices, self.A_dense]

        self.targets = _full_batch_tensor_and_reshape(targets, propagate_none=True)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.inputs, self.targets


class SparseFullBatchSequence(Sequence):
    """
    Keras-compatible data generator for for node inference models
    that require full-batch training (e.g., GCN, GAT).
    Use this class with the Keras methods :meth:`keras.Model.fit`,
        :meth:`keras.Model.evaluate`, and
        :meth:`keras.Model.predict`,

    This class uses sparse matrix representations to send data to the models,
    and only works with the Keras tensorflow backend. For any other backends,
    use the :class:`FullBatchSequence` class.

    This class should be created using the `.flow(...)` method of
    :class:`FullBatchNodeGenerator`.

    Args:
        features (np.ndarray): An array of node features of size (N x F),
            where N is the number of nodes in the graph, F is the node feature size
        A (sparse matrix): An adjacency matrix of the graph of size (N x N).
        targets (np.ndarray, optional): An optional array of node targets of size (N x C),
            where C is the target size (e.g., number of classes for one-hot class targets)
        indices (np.ndarray, optional): Array of indices to the feature and adjacency matrix
            of the targets. Required if targets is not None.
    """

    use_sparse = True

    def __init__(self, features, A, targets=None, indices=None):

        if (targets is not None) and (len(indices) != len(targets)):
            raise ValueError(
                "When passed together targets and indices should be the same length."
            )

        # Ensure matrix is in COO format to extract indices
        if sps.isspmatrix(A):
            A = A.tocoo()

        else:
            raise ValueError("Adjacency matrix not in expected sparse format")

        # Convert matrices to list of indices & values
        A_indices = np.hstack((A.row[:, None], A.col[:, None])).astype("int64")
        self.A_indices = _full_batch_tensor_and_reshape(A_indices)
        self.A_values = _full_batch_tensor_and_reshape(A.data)

        # Reshape all inputs to have batch dimension of 1
        self.target_indices = _full_batch_tensor_and_reshape(indices)
        self.features = _full_batch_tensor_and_reshape(features)
        self.inputs = [
            self.features,
            self.target_indices,
            self.A_indices,
            self.A_values,
        ]

        self.targets = _full_batch_tensor_and_reshape(targets, propagate_none=True)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.inputs, self.targets


class RelationalFullBatchNodeSequence(Sequence):
    """
    Keras-compatible data generator for for node inference models on relational graphs
    that require full-batch training (e.g., RGCN).
    Use this class with the Keras methods :meth:`keras.Model.fit`,
        :meth:`keras.Model.evaluate`, and
        :meth:`keras.Model.predict`,

    This class uses either dense or sparse representations to send data to the models.

    This class should be created using the `.flow(...)` method of
    :class:`RelationalFullBatchNodeGenerator`.

    Args:
        features (np.ndarray): An array of node features of size (N x F),
            where N is the number of nodes in the graph, F is the node feature size
        As (list of sparse matrices): A list of length R of adjacency matrices of the graph of size (N x N)
            where R is the number of relationships in the graph.
        targets (np.ndarray, optional): An optional array of node targets of size (N x C),
            where C is the target size (e.g., number of classes for one-hot class targets)
        indices (np.ndarray, optional): Array of indices to the feature and adjacency matrix
            of the targets. Required if targets is not None.
    """

    def __init__(self, features, As, use_sparse, targets=None, indices=None):

        if (targets is not None) and (len(indices) != len(targets)):
            raise ValueError(
                "When passed together targets and indices should be the same length."
            )

        self.use_sparse = use_sparse

        # Convert all adj matrices to dense and reshape to have batch dimension of 1
        if self.use_sparse:
            self.A_indices = [
                _full_batch_tensor_and_reshape(
                    np.hstack((A.row[:, None], A.col[:, None]))
                )
                for A in As
            ]
            self.A_values = [_full_batch_tensor_and_reshape(A.data) for A in As]
            self.As = self.A_indices + self.A_values
        else:
            self.As = [_full_batch_tensor_and_reshape(A.todense()) for A in As]

        # Make sure all inputs are numpy arrays, and have batch dimension of 1
        self.target_indices = _full_batch_tensor_and_reshape(indices)
        self.features = _full_batch_tensor_and_reshape(features)
        self.inputs = [self.features, self.target_indices] + self.As

        self.targets = _full_batch_tensor_and_reshape(targets, propagate_none=True)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.inputs, self.targets


class GraphSequence(Sequence):
    """
    A Keras-compatible data generator for training and evaluating graph classification models.
    Use this class with the Keras methods :meth:`keras.Model.fit`,
        :meth:`keras.Model.evaluate`, and
        :meth:`keras.Model.predict`,

    This class should be created using the `.flow(...)` method of
    :class:`GraphGenerator`.

    Args:
        graphs (list)): The graphs as StellarGraph objects.
        targets (np.ndarray, optional): An optional array of graph targets of size (N x C),
            where N is the number of graphs and C is the target size (e.g., number of classes.)
        normalize (bool, optional): Specifies whether the adjacency matrix for each graph should
            be normalized or not. The default is True.
        batch_size (int, optional): The batch size. It defaults to 1.
        name (str, optional): An optional name for this generator object.
    """

    def __init__(self, graphs, targets=None, normalize=True, batch_size=1, name=None):

        self.name = name
        self.graphs = np.asanyarray(graphs)
        self.normalize_adj = normalize
        self.targets = targets
        self.batch_size = batch_size

        if targets is not None:
            if len(graphs) != len(targets):
                raise ValueError(
                    "expected the number of target values and the number of graphs to be the same length,"
                    f"found {len(graphs)} graphs and {len(targets)} targets."
                )

            self.targets = np.asanyarray(targets)

        if self.normalize_adj:
            self.normalized_adjs = [
                normalize_adj(graph.to_adjacency_matrix()) for graph in graphs
            ]
        else:
            self.normalize_adjs = [graph.to_adjacency_matrix() for graph in graphs]

        self.normalized_adjs = np.asanyarray(self.normalized_adjs)

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.graphs) / self.batch_size))

    def __getitem__(self, index):

        batch_start, batch_end = index * self.batch_size, (index + 1) * self.batch_size

        graphs = self.graphs[batch_start:batch_end]
        adj_graphs = self.normalized_adjs[batch_start:batch_end]

        # The number of nodes for the largest graph in the batch. We are going to pad with 0 rows and columns
        # the adjacency and node feature matrices (only the rows in this case) to equal in size the adjacency and
        # feature matrices of the largest graph.
        max_nodes = max([graph.number_of_nodes() for graph in graphs])

        graph_targets = None
        if self.targets is not None:
            graph_targets = self.targets[batch_start:batch_end]

        # pad adjacency and feature matrices to equal the size of those from the largest graph
        with tf.device("/CPU:0"):
            features = [
                tf.pad(
                    graph.node_features_tensors(graph.nodes()),
                    ((0, max_nodes - graph.number_of_nodes()), (0, 0)),
                )
                for graph in graphs
            ]
            features = tf.stack(features)

        for adj in adj_graphs:
            adj.resize((max_nodes, max_nodes))
        adj_graphs = np.stack([adj.toarray() for adj in adj_graphs])

        masks = np.full((len(graphs), max_nodes), fill_value=False, dtype=np.bool)
        for index, graph in enumerate(graphs):
            masks[index, : graph.number_of_nodes()] = True

        # features is array of dimensionality
        #      batch size x N x F
        # masks is array of dimensionality
        #      batch size x N
        # adj_graphs is array of dimensionality
        #      batch size x N x N
        # graph_targets is array of dimensionality
        #      batch size x C
        # where N is the maximum number of nodes for largest graph in the batch, F is
        # the node feature dimensionality, and C is the number of target classes
        return [features, masks, adj_graphs], graph_targets

    def on_epoch_end(self):
        """
         Shuffle all graphs at the end of each epoch
        """
        indexes = list(range(len(self.graphs)))
        random.shuffle(indexes)
        self.graphs = self.graphs[indexes]
        self.normalized_adjs = self.normalized_adjs[indexes]
        if self.targets is not None:
            self.targets = self.targets[indexes]


class CorruptedNodeSequence(Sequence):
    """
    Keras compatible data generator that wraps a FullBatchSequence ot SparseFullBatchSequence and provides corrupted
    data for training Deep Graph Infomax.

    Args:
        base_generator: the uncorrupted Sequence object.
    """

    def __init__(self, base_generator):

        self.base_generator = base_generator

        if isinstance(base_generator, (FullBatchSequence, SparseFullBatchSequence)):
            batch_reps = (1, base_generator.target_indices.shape[1], 1)
            target = tf.constant([[[1.0, 0.0]]], dtype=tf.float32)
        elif isinstance(base_generator, NodeSequence):
            batch_reps = (base_generator.batch_size, 1)
            target = tf.constant([[1.0, 0.0]], dtype=tf.float32)
        else:
            raise TypeError(
                f"base_generator: expected FullBatchSequence, SparseFullBatchSequence, "
                f"or NodeSequence, found {type(base_generator).__name__}"
            )
        with tf.device("/CPU:0"):
            self.targets = tf.tile(target, batch_reps)

    def __len__(self):
        return len(self.base_generator)

    def __getitem__(self, index):

        inputs, _ = self.base_generator[index]

        with tf.device("/CPU:0"):

            if isinstance(
                self.base_generator, (FullBatchSequence, SparseFullBatchSequence)
            ):
                features = inputs[0]

                swap_batch = [1, 0, 2]
                nodes_first = tf.transpose(features, swap_batch)
                shuffled_nodes_first = tf.random.shuffle(nodes_first)
                shuffled_feats = [tf.transpose(shuffled_nodes_first, swap_batch)]
                targets = self.targets

            else:

                features = inputs
                feature_dim = tf.shape(features[0])[2]
                head_nodes = tf.shape(features[0])[0]

                stacked_feats = np.concatenate(
                    [tf.reshape(x, (-1, feature_dim)) for x in features], axis=0,
                )

                shuffled_feats = tf.random.shuffle(stacked_feats)
                shuffled_feats = tf.reshape(
                    shuffled_feats, (head_nodes, -1, feature_dim)
                )
                size_of_splits = [tf.shape(x)[1] for x in features]
                shuffled_feats = list(tf.split(shuffled_feats, size_of_splits, axis=1))

                targets = tf.squeeze(
                    tf.gather(self.targets, indices=[tf.range(head_nodes)]), axis=0
                )

        return shuffled_feats + inputs, targets
