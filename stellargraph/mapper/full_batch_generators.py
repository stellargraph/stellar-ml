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
    "FullBatchGenerator",
    "FullBatchNodeGenerator",
    "FullBatchLinkGenerator",
    "RelationalFullBatchNodeGenerator",
    "CorruptedGenerator",
]

import warnings
import operator
import random
import numpy as np
import itertools as it
import networkx as nx
import scipy.sparse as sps
from tensorflow.keras import backend as K
from functools import reduce
from tensorflow.keras.utils import Sequence

from . import (
    FullBatchSequence,
    SparseFullBatchSequence,
    RelationalFullBatchNodeSequence,
    CorruptedNodeSequence,
    GraphSAGENodeGenerator,
    DirectedGraphSAGENodeGenerator,
)
from ..core.graph import StellarGraph
from ..core.utils import is_real_iterable
from ..core.utils import GCN_Aadj_feats_op, PPNP_Aadj_feats_op

from abc import ABC


class FullBatchGenerator(ABC):
    multiplicity = None

    def __init__(
        self,
        G,
        name=None,
        method="gcn",
        k=1,
        sparse=True,
        transform=None,
        teleport_probability=0.1,
    ):
        if self.multiplicity is None:
            raise TypeError(
                "Can't instantiate abstract class 'FullBatchGenerator', please"
                "instantiate either 'FullBatchNodeGenerator' or 'FullBatchLinkGenerator'"
            )

        if not isinstance(G, StellarGraph):
            raise TypeError("Graph must be a StellarGraph or StellarDiGraph object.")

        self.graph = G
        self.name = name
        self.k = k
        self.teleport_probability = teleport_probability
        self.method = method

        # Check if the graph has features
        G.check_graph_for_ml()

        # Check that there is only a single node type for GAT or GCN
        node_types = list(G.node_types)
        if len(node_types) > 1:
            raise TypeError(
                "{}: node generator requires graph with single node type; "
                "a graph with multiple node types is passed. Stopping.".format(
                    type(self).__name__
                )
            )

        # Create sparse adjacency matrix:
        # Use the node orderings the same as in the graph features
        self.node_list = G.nodes()
        self.Aadj = G.to_adjacency_matrix(self.node_list)

        # Function to map node IDs to indices for quicker node index lookups
        # TODO: Move this to the graph class
        node_index_dict = dict(zip(self.node_list, range(len(self.node_list))))
        self._node_lookup = np.vectorize(node_index_dict.get, otypes=[np.int64])

        # Power-user feature: make the generator yield dense adjacency matrix instead
        # of the default sparse one.
        # If sparse is specified, check that the backend is tensorflow
        if sparse and K.backend() != "tensorflow":
            warnings.warn(
                "Sparse adjacency matrices are only supported in tensorflow."
                " Falling back to using a dense adjacency matrix."
            )
            self.use_sparse = False

        else:
            self.use_sparse = sparse

        # Get the features for the nodes
        self.features = G.node_features_tensors(self.node_list)

        if transform is not None:
            if callable(transform):
                self.features, self.Aadj = transform(
                    features=self.features, A=self.Aadj
                )
            else:
                raise ValueError("argument 'transform' must be a callable.")

        elif self.method in ["gcn", "sgc"]:
            self.features, self.Aadj = GCN_Aadj_feats_op(
                features=self.features, A=self.Aadj, k=self.k, method=self.method
            )

        elif self.method in ["gat", "self_loops"]:
            self.Aadj = self.Aadj + sps.diags(
                np.ones(self.Aadj.shape[0]) - self.Aadj.diagonal()
            )

        elif self.method in ["ppnp"]:
            if self.use_sparse:
                raise ValueError(
                    "use_sparse=true' is incompatible with 'ppnp'."
                    "Set 'use_sparse=True' or consider using the APPNP model instead."
                )
            self.features, self.Aadj = PPNP_Aadj_feats_op(
                features=self.features,
                A=self.Aadj,
                teleport_probability=self.teleport_probability,
            )

        elif self.method in [None, "none"]:
            pass

        else:
            raise ValueError(
                "Undefined method for adjacency matrix transformation. "
                "Accepted: 'gcn' (default), 'sgc', and 'self_loops'."
            )

    def flow(self, node_ids, targets=None):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        Args:
            node_ids: an iterable of node ids for the nodes of interest
                (e.g., training, validation, or test set nodes)
            targets: a 1D or 2D array of numeric node targets with shape `(len(node_ids)`
                or (len(node_ids), target_size)`

        Returns:
            A NodeSequence object to use with GCN or GAT models
            in Keras methods :meth:`fit`, :meth:`evaluate`,
            and :meth:`predict`

        """
        if targets is not None:
            # Check targets is an iterable
            if not is_real_iterable(targets):
                raise TypeError("Targets must be an iterable or None")

            # Check targets correct shape
            if len(targets) != len(node_ids):
                raise TypeError("Targets must be the same length as node_ids")

        # The list of indices of the target nodes in self.node_list
        node_indices = self._node_lookup(node_ids)

        if self.use_sparse:
            return SparseFullBatchSequence(
                self.features, self.Aadj, targets, node_indices
            )
        else:
            return FullBatchSequence(self.features, self.Aadj, targets, node_indices)


class FullBatchNodeGenerator(FullBatchGenerator):
    """
    A data generator for use with full-batch models on homogeneous graphs,
    e.g., GCN, GAT, SGC.
    The supplied graph G should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features to be available for all
    nodes in the graph.

    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    This generator will supply the features array and the adjacency matrix to a
    full-batch Keras graph ML model.  There is a choice to supply either a sparse
    adjacency matrix (the default) or a dense adjacency matrix, with the `sparse`
    argument.

    For these algorithms the adjacency matrix requires pre-processing and the
    'method' option should be specified with the correct pre-processing for
    each algorithm. The options are as follows:

    *   ``method='gcn'`` Normalizes the adjacency matrix for the GCN algorithm.
        This implements the linearized convolution of Eq. 8 in [1].
    *   ``method='sgc'``: This replicates the k-th order smoothed adjacency matrix
        to implement the Simplified Graph Convolutions of Eq. 8 in [2].
    *   ``method='self_loops'`` or ``method='gat'``: Simply sets the diagonal elements
        of the adjacency matrix to one, effectively adding self-loops to the graph. This is
        used by the GAT algorithm of [3].
    *   ``method='ppnp'`` Calculates the personalized page rank matrix of Eq 2 in [4].

    [1] `Kipf and Welling, 2017 <https://arxiv.org/abs/1609.02907>`_.
    [2] `Wu et al. 2019 <https://arxiv.org/abs/1902.07153>`_.
    [3] `Veličković et al., 2018 <https://arxiv.org/abs/1710.10903>`_
    [4] `Klicpera et al., 2018 <https://arxiv.org/abs/1810.05997>`_.

    Example::

        G_generator = FullBatchNodeGenerator(G)
        train_flow = G_generator.flow(node_ids, node_targets)

        # Fetch the data from train_flow, and feed into a Keras model:
        x_inputs, y_train = train_flow[0]
        model.fit(x=x_inputs, y=y_train)

        # Alternatively, use the generator itself with model.fit:
        model.fit(train_flow, epochs=num_epochs)

    For more information, please see the GCN, GAT, PPNP/APPNP and SGC demos: `<https://github.com/stellargraph/stellargraph/blob/master/demos/>`_

    Args:
        G (StellarGraphBase): a machine-learning StellarGraph-type graph
        name (str): an optional name of the generator
        method (str): Method to pre-process adjacency matrix. One of 'gcn' (default),
            'sgc', 'self_loops', or 'none'.
        k (None or int): This is the smoothing order for the 'sgc' method. This should be positive
            integer.
        transform (callable): an optional function to apply on features and adjacency matrix
            the function takes (features, Aadj) as arguments.
        sparse (bool): If True (default) a sparse adjacency matrix is used,
            if False a dense adjacency matrix is used.
        teleport_probability (float): teleport probability between 0.0 and 1.0.
            "probability" of returning to the starting node in the propagation step as in [4].
    """

    multiplicity = 1

    def flow(self, node_ids, targets=None):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        Args:
            node_ids: an iterable of node ids for the nodes of interest
                (e.g., training, validation, or test set nodes)
            targets: a 1D or 2D array of numeric node targets with shape `(len(node_ids)`
                or (len(node_ids), target_size)`

        Returns:
            A NodeSequence object to use with GCN or GAT models
            in Keras methods :meth:`fit`, :meth:`evaluate`,
            and :meth:`predict`

        """
        return super().flow(node_ids, targets)


class FullBatchLinkGenerator(FullBatchGenerator):
    """
    A data generator for use with full-batch models on homogeneous graphs,
    e.g., GCN, GAT, SGC.
    The supplied graph G should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features to be available for all
    nodes in the graph.

    Use the :meth:`flow` method supplying the links as a list of (src, dst) tuples
    of node IDs and (optionally) targets.

    This generator will supply the features array and the adjacency matrix to a
    full-batch Keras graph ML model.  There is a choice to supply either a sparse
    adjacency matrix (the default) or a dense adjacency matrix, with the `sparse`
    argument.

    For these algorithms the adjacency matrix requires pre-processing and the
    'method' option should be specified with the correct pre-processing for
    each algorithm. The options are as follows:

    *   ``method='gcn'`` Normalizes the adjacency matrix for the GCN algorithm.
        This implements the linearized convolution of Eq. 8 in [1].
    *   ``method='sgc'``: This replicates the k-th order smoothed adjacency matrix
        to implement the Simplified Graph Convolutions of Eq. 8 in [2].
    *   ``method='self_loops'`` or ``method='gat'``: Simply sets the diagonal elements
        of the adjacency matrix to one, effectively adding self-loops to the graph. This is
        used by the GAT algorithm of [3].
    *   ``method='ppnp'`` Calculates the personalized page rank matrix of Eq 2 in [4].

    [1] `Kipf and Welling, 2017 <https://arxiv.org/abs/1609.02907>`_.
    [2] `Wu et al. 2019 <https://arxiv.org/abs/1902.07153>`_.
    [3] `Veličković et al., 2018 <https://arxiv.org/abs/1710.10903>`_
    [4] `Klicpera et al., 2018 <https://arxiv.org/abs/1810.05997>`_.

    Example::

        G_generator = FullBatchLinkGenerator(G)
        train_flow = G_generator.flow([(1,2), (3,4), (5,6)], [0, 1, 1])

        # Fetch the data from train_flow, and feed into a Keras model:
        x_inputs, y_train = train_flow[0]
        model.fit(x=x_inputs, y=y_train)

        # Alternatively, use the generator itself with model.fit:
        model.fit(train_flow, epochs=num_epochs)

    For more information, please see the GCN, GAT, PPNP/APPNP and SGC demos: `<https://github.com/stellargraph/stellargraph/blob/master/demos/>`_

    Args:
        G (StellarGraphBase): a machine-learning StellarGraph-type graph
        name (str): an optional name of the generator
        method (str): Method to pre-process adjacency matrix. One of 'gcn' (default),
            'sgc', 'self_loops', or 'none'.
        k (None or int): This is the smoothing order for the 'sgc' method. This should be positive
            integer.
        transform (callable): an optional function to apply on features and adjacency matrix
            the function takes (features, Aadj) as arguments.
        sparse (bool): If True (default) a sparse adjacency matrix is used,
            if False a dense adjacency matrix is used.
        teleport_probability (float): teleport probability between 0.0 and 1.0. "probability"
            of returning to the starting node in the propagation step as in [4].
    """

    multiplicity = 2

    def flow(self, link_ids, targets=None):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        Args:
            link_ids: an iterable of link ids specified as tuples of node ids
                or an array of shape (N_links, 2) specifying the links.
            targets: a 1D or 2D array of numeric node targets with shape `(len(node_ids)`
                or (len(node_ids), target_size)`

        Returns:
            A NodeSequence object to use with GCN or GAT models
            in Keras methods :meth:`fit`, :meth:`evaluate`,
            and :meth:`predict`

        """
        return super().flow(link_ids, targets)


class RelationalFullBatchNodeGenerator:
    """
    A data generator for use with full-batch models on relational graphs e.g. RGCN.

    The supplied graph G should be a StellarGraph or StellarDiGraph object that is ready for
    machine learning. Currently the model requires node features to be available for all
    nodes in the graph.
    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    This generator will supply the features array and the adjacency matrix to a
    full-batch Keras graph ML model.  There is a choice to supply either a list of sparse
    adjacency matrices (the default) or a list of dense adjacency matrices, with the `sparse`
    argument.

    For these algorithms the adjacency matrices require pre-processing and the default option is to
    normalize each row of the adjacency matrix so that it sums to 1.
    For customization a transformation (callable) can be passed that
    operates on the node features and adjacency matrix.

    Example::

        G_generator = RelationalFullBatchNodeGenerator(G)
        train_data_gen = G_generator.flow(node_ids, node_targets)

        # Fetch the data from train_data_gen, and feed into a Keras model:
        # Alternatively, use the generator itself with model.fit:
        model.fit(train_gen, epochs=num_epochs, ...)

    Args:
        G (StellarGraph): a machine-learning StellarGraph-type graph
        name (str): an optional name of the generator
        transform (callable): an optional function to apply on features and adjacency matrix
            the function takes (features, Aadj) as arguments.
        sparse (bool): If True (default) a list of sparse adjacency matrices is used,
            if False a list of dense adjacency matrices is used.

    """

    def __init__(self, G, name=None, sparse=True, transform=None):

        if not isinstance(G, StellarGraph):
            raise TypeError("Graph must be a StellarGraph object.")

        self.graph = G
        self.name = name
        self.use_sparse = sparse
        self.multiplicity = 1

        # Check if the graph has features
        G.check_graph_for_ml()

        # extract node, feature, and edge type info from G
        self.node_list = list(G.nodes())

        self.features = G.node_features_tensors(self.node_list)

        edge_types = sorted(set(e[-1] for e in G.edges(include_edge_type=True)))
        self.node_index = dict(zip(self.node_list, range(len(self.node_list))))

        # create a list of adjacency matrices - one adj matrix for each edge type
        # an adjacency matrix is created for each edge type from all edges of that type
        self.As = []

        for edge_type in edge_types:

            col_index = [
                self.node_index[n1]
                for n1, n2, etype in G.edges(include_edge_type=True)
                if etype == edge_type
            ]
            row_index = [
                self.node_index[n2]
                for n1, n2, etype in G.edges(include_edge_type=True)
                if etype == edge_type
            ]
            data = np.ones(len(col_index), np.float64)

            # note that A is the transpose of the standard adjacency matrix
            # this is to aggregate features from incoming nodes
            A = sps.coo_matrix(
                (data, (row_index, col_index)),
                shape=(len(self.node_list), len(self.node_list)),
            )

            if transform is None:
                # normalize here and replace zero row sums with 1
                # to avoid harmless divide by zero warnings
                d = sps.diags(
                    np.float_power(np.ravel(np.maximum(A.sum(axis=1), 1)), -1), 0
                )
                A = d.dot(A)

            else:
                self.features, A = transform(self.features, A)

            A = A.tocoo()
            self.As.append(A)

    def flow(self, node_ids, targets=None):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        Args:
            node_ids: and iterable of node ids for the nodes of interest
                (e.g., training, validation, or test set nodes)
            targets: a 2D array of numeric node targets with shape `(len(node_ids), target_size)`

        Returns:
            A NodeSequence object to use with RGCN models
            in Keras methods :meth:`fit`, :meth:`evaluate`,
            and :meth:`predict`
        """

        if targets is not None:
            # Check targets is an iterable
            if not is_real_iterable(targets):
                raise TypeError("Targets must be an iterable or None")

            # Check targets correct shape
            if len(targets) != len(node_ids):
                raise TypeError("Targets must be the same length as node_ids")

        # The list of indices of the target nodes in self.node_list
        # use dictionary for faster index look-up time
        node_indices = np.array([self.node_index[n] for n in node_ids])

        return RelationalFullBatchNodeSequence(
            self.features, self.As, self.use_sparse, targets, node_indices
        )


class CorruptedGenerator:
    """
    Keras compatible data generator that wraps :class: `FullBatchNodeGenerator` and provides corrupted
    data for training Deep Graph Infomax.

    Args:
        base_generator: the uncorrupted Sequence object.
    """

    def __init__(self, base_generator):

        if not isinstance(
            base_generator,
            (
                FullBatchNodeGenerator,
                GraphSAGENodeGenerator,
                DirectedGraphSAGENodeGenerator,
            ),
        ):
            raise TypeError(
                f"base_generator: expected FullBatchNodeGenerator, GraphSAGENodeGenerator, "
                f"or DirectedGraphSAGENodeGenerator, found {type(base_generator).__name__}"
            )
        self.base_generator = base_generator

    def flow(self, *args, **kwargs):
        """
        Creates the corrupted :class: `Sequence` object for training Deep Graph Infomax.

        Args:
            args: the positional arguments for the self.base_generator.flow(...) method
            kwargs: the keyword arguments for the self.base_generator.flow(...) method
        """
        return CorruptedNodeSequence(self.base_generator.flow(*args, **kwargs))
