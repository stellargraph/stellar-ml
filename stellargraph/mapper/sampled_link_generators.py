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
Generators that create batches of data from a machine-learnign ready graph
for link prediction/link attribute inference problems using GraphSAGE, HinSAGE and Attri2Vec.

"""
__all__ = [
    "GraphSAGELinkGenerator",
    "HinSAGELinkGenerator",
    "Attri2VecLinkGenerator",
    "DirectedGraphSAGELinkGenerator",
]

import random
import operator
import numpy as np
import itertools as it
import operator
import collections
import abc
import warnings
from functools import reduce
from tensorflow import keras
from ..core.graph import StellarGraph, GraphSchema
from ..data import (
    SampledBreadthFirstWalk,
    SampledHeterogeneousBreadthFirstWalk,
    UniformRandomWalk,
    UnsupervisedSampler,
    DirectedBreadthFirstNeighbours,
)
from ..core.utils import is_real_iterable
import tensorflow as tf
from . import LinkSequence, OnDemandLinkSequence
from ..random import SeededPerBatch


class BatchedLinkGenerator(abc.ABC):
    def __init__(self, G, batch_size, schema=None):
        if not isinstance(G, StellarGraph):
            raise TypeError("Graph must be a StellarGraph or StellarDiGraph object.")

        self.graph = G
        self.batch_size = batch_size

        # This is a link generator and requries a model with two root nodes per query
        self.multiplicity = 2

        # Check if the graph has features
        G.check_graph_for_ml()

        # We need a schema for compatibility with HinSAGE
        if schema is None:
            self.schema = G.create_graph_schema()
        elif isinstance(schema, GraphSchema):
            self.schema = schema
        else:
            raise TypeError("Schema must be a GraphSchema object")

        # Do we need real node types here?
        self.head_node_types = None

        # Sampler (if required)
        self.sampler = None

    @abc.abstractmethod
    def sample_features(self, head_links, batch_num):
        pass

    def flow(self, link_ids, targets=None, shuffle=False, seed=None):
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
            link_ids: an iterable of tuples of node IDs as (source, target)
            targets: a 2D array of numeric targets with shape
                `(len(link_ids), target_size)`
            shuffle (bool): If True the links will be shuffled at each
                epoch, if False the links will be processed in order.
            seed (int, optional): Random seed

        Returns:
            A NodeSequence object to use with with StellarGraph models
            in Keras methods ``fit``, ``evaluate``,
            and ``predict``

        """
        if self.head_node_types is not None:
            expected_src_type = self.head_node_types[0]
            expected_dst_type = self.head_node_types[1]

        # Pass sampler to on-demand link sequence generation
        if isinstance(link_ids, UnsupervisedSampler):
            return OnDemandLinkSequence(self.sample_features, self.batch_size, link_ids)

        # Otherwise pass iterable (check?) to standard LinkSequence
        elif isinstance(link_ids, collections.abc.Iterable):
            # Check all IDs are actually in the graph and are of expected type
            for link in link_ids:
                if len(link) != 2:
                    raise KeyError("Expected link IDs to be a tuple of length 2")

                src, dst = link
                try:
                    node_type_src = self.graph.node_type(src)
                except KeyError:
                    raise KeyError(
                        f"Node ID {src} supplied to generator not found in graph"
                    )
                try:
                    node_type_dst = self.graph.node_type(dst)
                except KeyError:
                    raise KeyError(
                        f"Node ID {dst} supplied to generator not found in graph"
                    )

                if self.head_node_types is not None and (
                    node_type_src != expected_src_type
                    or node_type_dst != expected_dst_type
                ):
                    raise ValueError(
                        f"Node pair ({src}, {dst}) not of expected type ({expected_src_type}, {expected_dst_type})"
                    )

            return LinkSequence(
                self.sample_features,
                self.batch_size,
                link_ids,
                targets=targets,
                shuffle=shuffle,
                seed=seed,
            )

        else:
            raise TypeError(
                "Argument to .flow not recognised. "
                "Please pass a list of samples or a UnsupervisedSampler object."
            )

    def flow_from_dataframe(self, link_targets, shuffle=False):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        Args:
            link_targets: a Pandas DataFrame of links specified by
                'source' and 'target' and an optional target label
                specified by 'label'.
            shuffle (bool): If True the links will be shuffled at each
                epoch, if False the links will be processed in order.

        Returns:
            A NodeSequence object to use with StellarGraph models
            in Keras methods ``fit``, ``evaluate``,
            and ``predict``

        """
        return self.flow(
            link_targets["source", "target"].values,
            link_targets["label"].values,
            shuffle=shuffle,
        )


class GraphSAGELinkGenerator(BatchedLinkGenerator):
    """
    A data generator for link prediction with Homogeneous GraphSAGE models

    At minimum, supply the StellarGraph, the batch size, and the number of
    node samples for each layer of the GraphSAGE model.

    The supplied graph should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Use the :meth:`.flow` method supplying the nodes and (optionally) targets,
    or an UnsupervisedSampler instance that generates node samples on demand,
    to get an object that can be used as a Keras data generator.

    Example::

        G_generator = GraphSageLinkGenerator(G, 50, [10,10])
        train_data_gen = G_generator.flow(edge_ids)

    Args:
        G (StellarGraph): A machine-learning ready graph.
        batch_size (int): Size of batch of links to return.
        num_samples (list): List of number of neighbour node samples per GraphSAGE layer (hop) to take.
        seed (int or str), optional: Random seed for the sampling methods.
    """

    def __init__(self, G, batch_size, num_samples, seed=None, name=None):
        super().__init__(G, batch_size)

        self.num_samples = num_samples
        self.name = name

        # Check that there is only a single node type for GraphSAGE
        if len(self.schema.node_types) > 1:
            warnings.warn(
                "running homogeneous GraphSAGE on a graph with multiple node types",
                RuntimeWarning,
                stacklevel=2,
            )

        self.head_node_types = self.schema.node_types * 2

        self._graph = G
        self._samplers = SeededPerBatch(
            lambda s: SampledBreadthFirstWalk(
                self._graph, graph_schema=self.schema, seed=s
            ),
            seed=seed,
        )

    def sample_features(self, head_links, batch_num):
        """
        Sample neighbours recursively from the head nodes, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_links: An iterable of edges to perform sampling for.
            batch_num (int): Batch number

        Returns:
            A list of the same length as ``num_samples`` of collected features from
            the sampled nodes of shape:
            ``(len(head_nodes), num_sampled_at_layer, feature_size)``
            where num_sampled_at_layer is the cumulative product of `num_samples`
            for that layer.
        """
        node_type = self.head_node_types[0]
        head_size = len(head_links)

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

        # Get sampled nodes for the subgraphs for the edges where each edge is a tuple
        # of 2 nodes, so we are extracting 2 head nodes per edge
        with tf.device("/CPU:0"):
            batch_feats = []
            for hns in zip(*head_links):
                node_samples = self._samplers[batch_num].run(
                    nodes=hns, n=1, n_size=self.num_samples
                )

                nodes_per_hop = get_levels(0, 1, self.num_samples, node_samples)

                # Get features for the sampled nodes
                batch_feats.append(
                    [
                        self.graph.node_features_tensors(layer_nodes, node_type)
                        for layer_nodes in nodes_per_hop
                    ]
                )

            # Resize features to (batch_size, n_neighbours, feature_size)
            # and re-pack features into a list where source, target feats alternate
            # This matches the GraphSAGE link model with (node_src, node_dst) input sockets:
            batch_feats = [
                tf.reshape(feats, (head_size, -1, feats.shape[1]))
                for ab in zip(*batch_feats)
                for feats in ab
            ]
        return batch_feats


class HinSAGELinkGenerator(BatchedLinkGenerator):
    """
    A data generator for link prediction with Heterogeneous HinSAGE models

    At minimum, supply the StellarGraph, the batch size, and the number of
    node samples for each layer of the GraphSAGE model.

    The supplied graph should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    The generator should be given the (src,dst) node types usng

    * It's possible to do link prediction on a graph where that link type is completely removed from the graph
      (e.g., "same_as" links in ER)


    Args:
        g (StellarGraph): A machine-learning ready graph.
        batch_size (int): Size of batch of links to return.
        num_samples (list): List of number of neighbour node samples per GraphSAGE layer (hop) to take.
        head_node_types (list): List of the types (str) of the two head nodes forming the node pair.
        seed (int or str, optional): Random seed for the sampling methods.

    Example::

        G_generator = HinSAGELinkGenerator(G, 50, [10,10])
        data_gen = G_generator.flow(edge_ids)
    """

    def __init__(
        self,
        G,
        batch_size,
        num_samples,
        head_node_types,
        schema=None,
        seed=None,
        name=None,
    ):
        super().__init__(G, batch_size, schema)
        self.num_samples = num_samples
        self.name = name

        # This is a link generator and requires two nodes per query
        self.head_node_types = head_node_types
        if len(self.head_node_types) != 2:
            raise ValueError(
                "The head_node_types should be of length 2 for a link generator"
            )

        # Create sampling schema
        self._sampling_schema = self.schema.sampling_layout(
            self.head_node_types, self.num_samples
        )
        self._type_adjacency_list = self.schema.type_adjacency_list(
            self.head_node_types, len(self.num_samples)
        )

        # The sampler used to generate random samples of neighbours
        self.sampler = SampledHeterogeneousBreadthFirstWalk(
            G, graph_schema=self.schema, seed=seed
        )

    def _get_features(self, node_samples, head_size):
        """
        Collect features from sampled nodes.
        Args:
            node_samples: A list of lists of node IDs
            head_size: The number of head nodes (typically the batch size).

        Returns:
            A list of numpy arrays that store the features for each head
            node.
        """
        # Note the if there are no samples for a node a zero array is returned.
        # Resize features to (batch_size, n_neighbours, feature_size)
        # for each node type (note that we can have different feature size for each node type)
        with tf.device("/CPU:0"):

            batch_feats = [
                self.graph.node_features_tensors(layer_nodes, nt)
                for nt, layer_nodes in node_samples
            ]

            # Resize features to (batch_size, n_neighbours, feature_size)
            batch_feats = [
                tf.reshape(a, (head_size, -1, a.shape[1])) for a in batch_feats
            ]

        return batch_feats

    def sample_features(self, head_links, batch_num):
        """
        Sample neighbours recursively from the head nodes, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_links (list): An iterable of edges to perform sampling for.
            batch_num (int): Batch number

        Returns:
            A list of the same length as `num_samples` of collected features from
            the sampled nodes of shape: `(len(head_nodes), num_sampled_at_layer, feature_size)`
            where num_sampled_at_layer is the cumulative product of `num_samples`
            for that layer.
        """
        nodes_by_type = []
        for ii in range(2):
            # Extract head nodes from edges: each edge is a tuple of 2 nodes, so we are extracting 2 head nodes per edge
            head_nodes = [e[ii] for e in head_links]

            # Get sampled nodes for the subgraphs starting from the (src, dst) head nodes
            # nodes_samples is list of two lists: [[samples for src], [samples for dst]]
            node_samples = self.sampler.run(
                nodes=head_nodes, n=1, n_size=self.num_samples
            )

            # Reshape node samples to the required format for the HinSAGE model
            # This requires grouping the sampled nodes by edge type and in order
            nodes_by_type.append(
                [
                    (
                        nt,
                        reduce(
                            operator.concat,
                            (samples[ks] for samples in node_samples for ks in indices),
                            [],
                        ),
                    )
                    for nt, indices in self._sampling_schema[ii]
                ]
            )

        # Interlace the two lists, nodes_by_type[0] (for src head nodes) and nodes_by_type[1] (for dst head nodes)
        nodes_by_type = [
            tuple((ab[0][0], reduce(operator.concat, (ab[0][1], ab[1][1]))))
            for ab in zip(nodes_by_type[0], nodes_by_type[1])
        ]

        batch_feats = self._get_features(nodes_by_type, len(head_links))

        return batch_feats


class Attri2VecLinkGenerator(BatchedLinkGenerator):
    """
    A data generator for context node prediction with the attri2vec model.

    At minimum, supply the StellarGraph and the batch size.

    The supplied graph should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Use the :meth:`.flow` method supplying the nodes and targets,
    or an UnsupervisedSampler instance that generates node samples on demand,
    to get an object that can be used as a Keras data generator.

    Example::

        G_generator = Attri2VecLinkGenerator(G, 50)
        train_data_gen = G_generator.flow(edge_ids, edge_labels)

    Args:
        G (StellarGraph): A machine-learning ready graph.
        batch_size (int): Size of batch of links to return.
        name, optional: Name of generator.
    """

    def __init__(self, G, batch_size, name=None):
        super().__init__(G, batch_size)

        self.name = name

    def sample_features(self, head_links, batch_num):
        """
        Sample content features of the target nodes and the ids of the context nodes
        and return these as a list of feature arrays for the attri2vec algorithm.

        Args:
            head_links: An iterable of edges to perform sampling for.
            batch_num (int): Batch number

        Returns:
            A list of feature arrays, with each element being the feature of a
            target node and the id of the corresponding context node.
        """

        target_ids = [head_link[0] for head_link in head_links]
        context_ids = [head_link[1] for head_link in head_links]
        target_feats = self.graph.node_features_tensors(target_ids)
        context_feats = self.graph._get_index_for_nodes(context_ids)
        batch_feats = [target_feats, np.array(context_feats)]

        return batch_feats


class DirectedGraphSAGELinkGenerator(BatchedLinkGenerator):
    """
    A data generator for link prediction with directed Homogeneous GraphSAGE models

    At minimum, supply the StellarDiGraph, the batch size, and the number of
    node samples (separately for in-nodes and out-nodes) for each layer of the GraphSAGE model.

    The supplied graph should be a StellarDiGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Use the :meth:`.flow` method supplying the nodes and (optionally) targets,
    or an UnsupervisedSampler instance that generates node samples on demand,
    to get an object that can be used as a Keras data generator.

    Example::

        G_generator = DirectedGraphSageLinkGenerator(G, 50, [10,10], [10,10])
        train_data_gen = G_generator.flow(edge_ids)

    Args:
        G (StellarGraph): A machine-learning ready graph.
        batch_size (int): Size of batch of links to return.
        in_samples (list): The number of in-node samples per layer (hop) to take.
        out_samples (list): The number of out-node samples per layer (hop) to take.
        seed (int or str), optional: Random seed for the sampling methods.
        name, optional: Name of generator.
    """

    def __init__(self, G, batch_size, in_samples, out_samples, seed=None, name=None):
        super().__init__(G, batch_size)

        self.in_samples = in_samples
        self.out_samples = out_samples
        self._name = name

        # Check that there is only a single node type for GraphSAGE
        if len(self.schema.node_types) > 1:
            warnings.warn(
                "running homogeneous GraphSAGE on a graph with multiple node types",
                RuntimeWarning,
                stacklevel=2,
            )

        self.head_node_types = self.schema.node_types * 2

        self._graph = G

        self._samplers = SeededPerBatch(
            lambda s: DirectedBreadthFirstNeighbours(
                self._graph, graph_schema=self.schema, seed=s
            ),
            seed=seed,
        )

    def sample_features(self, head_links, batch_num):
        """
        Sample neighbours recursively from the head links, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_links: An iterable of head links to perform sampling on.

        Returns:
            A list of feature tensors from the sampled nodes at each layer, each of shape:
            ``(len(head_nodes), num_sampled_at_layer, feature_size)``
            where num_sampled_at_layer is the total number (cumulative product)
            of nodes sampled at the given number of hops from each head node,
            given the sequence of in/out directions.
        """

        batch_feats = []
        for hns in zip(*head_links):

            node_samples = self._samplers[batch_num].run(
                nodes=hns, n=1, in_size=self.in_samples, out_size=self.out_samples
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
                    nodes_in_slot = [
                        element for sample in node_samples for element in sample[slot]
                    ]
                    features_for_slot = self.graph.node_features_tensors(
                        nodes_in_slot, node_type
                    )

                    features[slot] = tf.reshape(
                        features_for_slot, (len(hns), -1, features_for_slot.shape[1])
                    )

            # Get features for the sampled nodes
            batch_feats.append(features)

        # Resize features to (batch_size, n_neighbours, feature_size)
        # and re-pack features into a list where source, target feats alternate
        # This matches the GraphSAGE link model with (node_src, node_dst) input sockets:
        batch_feats = [feats for ab in zip(*batch_feats) for feats in ab]
        return batch_feats
