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
__all__ = [
    "GraphSAGENodeGenerator",
    "HinSAGENodeGenerator",
    "Attri2VecNodeGenerator",
    "DirectedGraphSAGENodeGenerator",
]

import warnings
import operator
import random
import abc
import warnings
import numpy as np
import itertools as it
import networkx as nx
import scipy.sparse as sps
from tensorflow.keras import backend as K
from functools import reduce
from tensorflow.keras.utils import Sequence
import tensorflow as tf

from ..data import (
    SampledBreadthFirstWalk,
    SampledHeterogeneousBreadthFirstWalk,
    DirectedBreadthFirstNeighbours,
)
from ..core.graph import StellarGraph, GraphSchema
from ..core.utils import is_real_iterable
from . import NodeSequence
from ..random import SeededPerBatch


class BatchedNodeGenerator(abc.ABC):
    """
    Abstract base class for graph data generators.

    The supplied graph should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Do not use this base class: use a subclass specific to the method.

    Args:
        G (StellarGraph): The machine-learning ready graph.
        batch_size (int): Size of batch to return.
        schema (GraphSchema): [Optional] Schema for the graph, for heterogeneous graphs.
    """

    def __init__(self, G, batch_size, schema=None):
        if not isinstance(G, StellarGraph):
            raise TypeError("Graph must be a StellarGraph or StellarDiGraph object.")

        self.graph = G
        self.batch_size = batch_size

        # This is a node generator and requries a model with one root nodes per query
        self.multiplicity = 1

        # Check if the graph has features
        G.check_graph_for_ml()

        # We need a schema for compatibility with HinSAGE
        if schema is None:
            self.schema = G.create_graph_schema()
        elif isinstance(schema, GraphSchema):
            self.schema = schema
        else:
            raise TypeError("Schema must be a GraphSchema object")

        # We will need real node types here
        self.head_node_types = None

        # Create sampler for GraphSAGE
        self.sampler = None

    @abc.abstractmethod
    def sample_features(self, head_nodes, batch_num):
        pass

    def flow(self, node_ids, targets=None, shuffle=False, seed=None):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        The node IDs are the nodes to train or inference on: the embeddings
        calculated for these nodes are passed to the downstream task. These
        are a subset of the nodes in the graph.

        The targets are an array of numeric targets corresponding to the
        supplied node_ids to be used by the downstream task. They should
        be given in the same order as the list of node IDs.
        If they are not specified (for example, for use in prediction),
        the targets will not be available to the downstream task.

        Note that the shuffle argument should be True for training and
        False for prediction.

        Args:
            node_ids: an iterable of node IDs
            targets: a 2D array of numeric targets with shape
                `(len(node_ids), target_size)`
            shuffle (bool): If True the node_ids will be shuffled at each
                epoch, if False the node_ids will be processed in order.

        Returns:
            A NodeSequence object to use with with StellarGraph models
            in Keras methods ``fit``, ``evaluate``,
            and ``predict``

        """
        if self.head_node_types is not None:
            expected_node_type = self.head_node_types[0]
        else:
            expected_node_type = None

        # Check all IDs are actually in the graph and are of expected type
        for n in node_ids:
            try:
                node_type = self.graph.node_type(n)
            except KeyError:
                raise KeyError(f"Node ID {n} supplied to generator not found in graph")

            if expected_node_type is not None and (node_type != expected_node_type):
                raise ValueError(
                    f"Node ID {n} not of expected type {expected_node_type}"
                )

        return NodeSequence(
            self.sample_features,
            self.batch_size,
            node_ids,
            targets,
            shuffle=shuffle,
            seed=seed,
        )

    def flow_from_dataframe(self, node_targets, shuffle=False):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        Args:
            node_targets: a Pandas DataFrame of numeric targets indexed
                by the node ID for that target.
            shuffle (bool): If True the node_ids will be shuffled at each
                epoch, if False the node_ids will be processed in order.

        Returns:
            A NodeSequence object to use with with StellarGraph models
            in Keras methods ``fit``, ``evaluate``,
            and ``predict``

        """
        return self.flow(node_targets.index, node_targets.values, shuffle=shuffle)


class GraphSAGENodeGenerator(BatchedNodeGenerator):
    """
    A data generator for node prediction with Homogeneous GraphSAGE models

    At minimum, supply the StellarGraph, the batch size, and the number of
    node samples for each layer of the GraphSAGE model.

    The supplied graph should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    Example::

        G_generator = GraphSAGENodeGenerator(G, 50, [10,10])
        train_data_gen = G_generator.flow(train_node_ids, train_node_labels)
        test_data_gen = G_generator.flow(test_node_ids)

    Args:
        G (StellarGraph): The machine-learning ready graph.
        batch_size (int): Size of batch to return.
        num_samples (list): The number of samples per layer (hop) to take.
        seed (int): [Optional] Random seed for the node sampler.
    """

    def __init__(self, G, batch_size, num_samples, seed=None, name=None):
        super().__init__(G, batch_size)

        self.num_samples = num_samples
        self.head_node_types = self.schema.node_types
        self.name = name

        # Check that there is only a single node type for GraphSAGE
        if len(self.head_node_types) > 1:
            warnings.warn(
                "running homogeneous GraphSAGE on a graph with multiple node types",
                RuntimeWarning,
                stacklevel=2,
            )

        # Create sampler for GraphSAGE
        self._samplers = SeededPerBatch(
            lambda s: SampledBreadthFirstWalk(G, graph_schema=self.schema, seed=s),
            seed=seed,
        )

    def sample_features(self, head_nodes, batch_num):
        """
        Sample neighbours recursively from the head nodes, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_nodes: An iterable of head nodes to perform sampling on.
            batch_num (int): Batch number

        Returns:
            A list of the same length as ``num_samples`` of collected features from
            the sampled nodes of shape:
            ``(len(head_nodes), num_sampled_at_layer, feature_size)``
            where num_sampled_at_layer is the cumulative product of `num_samples`
            for that layer.
        """
        node_samples = self._samplers[batch_num].run(
            nodes=head_nodes, n=1, n_size=self.num_samples
        )

        # The number of samples for each head node (not including itself)
        num_full_samples = np.sum(np.cumprod(self.num_samples))

        # Reshape node samples to sensible format
        def get_levels(loc, lsize, samples_per_hop, walks):
            end_loc = loc + lsize
            walks_at_level = list(it.chain(*[w[loc:end_loc] for w in walks]))
            if len(samples_per_hop) < 1:
                return [walks_at_level]
            return [walks_at_level] + get_levels(
                end_loc, lsize * samples_per_hop[0], samples_per_hop[1:], walks
            )

        nodes_per_hop = get_levels(0, 1, self.num_samples, node_samples)
        node_type = self.head_node_types[0]

        with tf.device("/CPU:0"):
            # Get features for sampled nodes
            batch_feats = [
                self.graph.node_features_tensors(layer_nodes, node_type)
                for layer_nodes in nodes_per_hop
            ]

            # Resize features to (batch_size, n_neighbours, feature_size)
            batch_feats = [
                tf.reshape(a, (len(head_nodes), -1, a.shape[1])) for a in batch_feats
            ]
        return batch_feats


class DirectedGraphSAGENodeGenerator(BatchedNodeGenerator):
    """
    A data generator for node prediction with homogeneous GraphSAGE models
    on directed graphs.

    At minimum, supply the StellarDiGraph, the batch size, and the number of
    node samples (separately for in-nodes and out-nodes)
    for each layer of the GraphSAGE model.

    The supplied graph should be a StellarDiGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    Example::

        G_generator = DirectedGraphSAGENodeGenerator(G, 50, [10,5], [5,1])
        train_data_gen = G_generator.flow(train_node_ids, train_node_labels)
        test_data_gen = G_generator.flow(test_node_ids)

    Args:
        G (StellarDiGraph): The machine-learning ready graph.
        batch_size (int): Size of batch to return.
        in_samples (list): The number of in-node samples per layer (hop) to take.
        out_samples (list): The number of out-node samples per layer (hop) to take.
        seed (int): [Optional] Random seed for the node sampler.
    """

    def __init__(self, G, batch_size, in_samples, out_samples, seed=None, name=None):
        super().__init__(G, batch_size)

        # TODO Add checks for in- and out-nodes sizes
        self.in_samples = in_samples
        self.out_samples = out_samples
        self.head_node_types = self.schema.node_types
        self.name = name

        # Check that there is only a single node type for GraphSAGE
        if len(self.head_node_types) > 1:
            warnings.warn(
                "running homogeneous GraphSAGE on a graph with multiple node types",
                RuntimeWarning,
                stacklevel=2,
            )

        # Create sampler for GraphSAGE
        self.sampler = DirectedBreadthFirstNeighbours(
            G, graph_schema=self.schema, seed=seed
        )

    def sample_features(self, head_nodes, batch_num):
        """
        Sample neighbours recursively from the head nodes, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_nodes: An iterable of head nodes to perform sampling on.
            batch_num (int): Batch number

        Returns:
            A list of feature tensors from the sampled nodes at each layer, each of shape:
            ``(len(head_nodes), num_sampled_at_layer, feature_size)``
            where num_sampled_at_layer is the total number (cumulative product)
            of nodes sampled at the given number of hops from each head node,
            given the sequence of in/out directions.
        """
        node_samples = self.sampler.run(
            nodes=head_nodes, n=1, in_size=self.in_samples, out_size=self.out_samples
        )

        # Reshape node samples to sensible format
        # Each 'slot' represents the list of nodes sampled from some neighbourhood, and will have a corresponding
        # NN input layer. Every hop potentially generates both in-nodes and out-nodes, held separately,
        # and thus the slot (or directed hop sequence) structure forms a binary tree.

        node_type = self.head_node_types[0]

        max_hops = len(self.in_samples)
        max_slots = 2 ** (max_hops + 1) - 1
        features = [None] * max_slots  # flattened binary tree

        with tf.device("/CPU:0"):
            for slot in range(max_slots):
                nodes_in_slot = list(
                    it.chain(*[sample[slot] for sample in node_samples])
                )
                features_for_slot = self.graph.node_features_tensors(
                    nodes_in_slot, node_type
                )
                features[slot] = tf.reshape(
                    features_for_slot, (len(head_nodes), -1, features_for_slot.shape[1])
                )

        return features


class HinSAGENodeGenerator(BatchedNodeGenerator):
    """Keras-compatible data mapper for Heterogeneous GraphSAGE (HinSAGE)

    At minimum, supply the StellarGraph, the batch size, and the number of
    node samples for each layer of the HinSAGE model.

    The supplied graph should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    Note that the shuffle argument should be True for training and
    False for prediction.

    Args:
        G (StellarGraph): The machine-learning ready graph
        batch_size (int): Size of batch to return
        num_samples (list): The number of samples per layer (hop) to take
        head_node_type (str): The node type that will be given to the generator
            using the `flow` method, the model will expect this node type.
        schema (GraphSchema, optional): Graph schema for G.
        seed (int, optional): Random seed for the node sampler

    Example::

         G_generator = HinSAGENodeGenerator(G, 50, [10,10])
         train_data_gen = G_generator.flow(train_node_ids, train_node_labels)
         test_data_gen = G_generator.flow(test_node_ids)

     """

    def __init__(
        self,
        G,
        batch_size,
        num_samples,
        head_node_type,
        schema=None,
        seed=None,
        name=None,
    ):
        super().__init__(G, batch_size, schema=schema)

        self.num_samples = num_samples
        self.name = name

        # The head node type
        if head_node_type not in self.schema.node_types:
            raise KeyError("Supplied head node type must exist in the graph")
        self.head_node_types = [head_node_type]

        # Create sampling schema
        self._sampling_schema = self.schema.sampling_layout(
            self.head_node_types, self.num_samples
        )
        self._type_adjacency_list = self.schema.type_adjacency_list(
            self.head_node_types, len(self.num_samples)
        )

        # Create sampler for HinSAGE
        self.sampler = SampledHeterogeneousBreadthFirstWalk(
            G, graph_schema=self.schema, seed=seed
        )

    def sample_features(self, head_nodes, batch_num):
        """
        Sample neighbours recursively from the head nodes, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_nodes: An iterable of head nodes to perform sampling on.
            batch_num (int): Batch number

        Returns:
            A list of the same length as ``num_samples`` of collected features from
            the sampled nodes of shape:
            ``(len(head_nodes), num_sampled_at_layer, feature_size)``
            where num_sampled_at_layer is the cumulative product of `num_samples`
            for that layer.
        """
        # Get sampled nodes
        node_samples = self.sampler.run(nodes=head_nodes, n=1, n_size=self.num_samples)

        # Reshape node samples to the required format for the HinSAGE model
        # This requires grouping the sampled nodes by edge type and in order
        nodes_by_type = [
            (
                nt,
                reduce(
                    operator.concat,
                    (samples[ks] for samples in node_samples for ks in indices),
                    [],
                ),
            )
            for nt, indices in self._sampling_schema[0]
        ]

        with tf.device("/CPU:0"):
            # Get features
            batch_feats = [
                self.graph.node_features_tensors(layer_nodes, nt)
                for nt, layer_nodes in nodes_by_type
            ]

            # Resize features to (batch_size, n_neighbours, feature_size)
            batch_feats = [
                tf.reshape(a, (len(head_nodes), -1, a.shape[1])) for a in batch_feats
            ]

        return batch_feats


class Attri2VecNodeGenerator(BatchedNodeGenerator):
    """
    A node feature generator for node representation prediction with the
    attri2vec model.

    At minimum, supply the StellarGraph and the batch size.

    The supplied graph should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Use the :meth:`flow` method supplying the nodes to get an object
    that can be used as a Keras data generator.

    Example::

        G_generator = Attri2VecNodeGenerator(G, 50)
        data_gen = G_generator.flow(node_ids)

    Args:
        G (StellarGraph): The machine-learning ready graph.
        batch_size (int): Size of batch to return.
        name (str or None): Name of the generator (optional).
    """

    def __init__(self, G, batch_size, name=None):
        super().__init__(G, batch_size)
        self.name = name

    def sample_features(self, head_nodes, batch_num):
        """
        Sample content features of the head nodes, and return these as a list of feature
        arrays for the attri2vec algorithm.

        Args:
            head_nodes: An iterable of head nodes to perform sampling on.
            batch_num (int): Batch number

        Returns:
            A list of feature arrays, with each element being the feature of a
            head node.
        """

        batch_feats = self.graph.node_features_tensors(head_nodes)
        return batch_feats

    def flow(self, node_ids):
        """
        Creates a generator/sequence object for node representation prediction
        with the supplied node ids.

        The node IDs are the nodes to inference on: the embeddings
        calculated for these nodes are passed to the downstream task. These
        are a subset/all of the nodes in the graph.

        Args:
            node_ids: an iterable of node IDs.

        Returns:
            A NodeSequence object to use with the Attri2Vec model
            in the Keras method ``predict``.

        """
        return NodeSequence(
            self.sample_features, self.batch_size, node_ids, shuffle=False
        )

    def flow_from_dataframe(self, node_ids):
        """
        Creates a generator/sequence object for node representation prediction
        with the supplied node ids.

        Args:
            node_ids: a Pandas DataFrame of node_ids.

        Returns:
            A NodeSequence object to use with the Attri2Vec model
            in the Keras method ``predict``.

        """
        return NodeSequence(self.sample_features, self.batch_size, node_ids.index)
