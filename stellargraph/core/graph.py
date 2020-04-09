# -*- coding: utf-8 -*-
#
# Copyright 2017-2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The StellarGraph class that encapsulates information required for
a machine-learning ready graph used by models.

"""
__all__ = ["StellarGraph", "StellarDiGraph", "GraphSchema", "NeighbourWithWeight"]

from typing import Iterable, Any, Mapping, List, Optional, Set
from collections import defaultdict, namedtuple
import pandas as pd
import numpy as np
import scipy.sparse as sps
import warnings
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from .. import globalvar
from .schema import GraphSchema, EdgeType
from .experimental import experimental, ExperimentalWarning
from .element_data import NodeData, EdgeData, ExternalIdIndex
from .utils import is_real_iterable
from .validation import comma_sep, separated
from . import convert


NeighbourWithWeight = namedtuple("NeighbourWithWeight", ["node", "weight"])


class StellarGraph:
    """
    StellarGraph class for graph machine learning.

    Summary of a StellarGraph and the terminology used:

    - it stores graph structure, as a collection of *nodes* and a collection of *edges* that connect
      a *source* node to a *target* node

    - each node and edge has an associated *type*

    - each node has a numeric vector of *features*, and the vectors of all nodes with the same type
      have the same dimension

    - it is *homogeneous* if there is only one type of node and one type of edge

    - it is *heterogeneous* if it is not homgeneous (more than one type of node, or more than
      one type of edge)

    - it is *directed* if the direction of an edge starting at its source node and finishing at
      its target node is important

    - it is *undirected* if the direction does not matter

    - every StellarGraph can be a *multigraph*, meaning there can be multiple edges between any two
      nodes

    To create a StellarGraph object, at a minimum pass the edges as a Pandas
    DataFrame. Each row of the edges DataFrame represents an edge, where the index is the
    ID of the edge, and the ``source`` and ``target`` columns store the node ID of the source and
    target nodes.

    For example, suppose we're modelling a graph that's a square with a diagonal::

        a -- b
        | \\  |
        |  \\ |
        d -- c

    The DataFrame might look like::

        edges = pd.DataFrame(
            {"source": ["a", "b", "c", "d", "a"], "target": ["b", "c", "d", "a", "c"]}
        )

    If this data represents an undirected graph (the ordering of each edge source/target doesn't
    matter)::

        Gs = StellarGraph(edges=edges)

    If this data represents a directed graph (the ordering does matter)::

        Gs = StellarDiGraph(edges=edges)

    One can also pass a DataFrame of nodes. Each row of the nodes DataFrame represents a node in the
    graph, where the index is the ID of the node. When this nodes DataFrame is not passed (the
    argument is left as the default), the set of nodes is automatically inferred. This inference in
    the example above is equivalent to::

        nodes = pd.DataFrame([], index=["a", "b", "c", "d"])
        Gs = StellarGraph(nodes, edges)

    Numeric node features are taken as any columns of the nodes DataFrame. For example, if the graph
    above has two features ``x`` and ``y`` associated with each node::

        nodes = pd.DataFrame(
            {"x": [-1, 2, -3, 4], "y": [0.4, 0.1, 0.9, 0]}, index=["a", "b", "c", "d"]
        )

    Edge weights are taken as the optional ``weight`` column of the edges DataFrame::

        edges = pd.DataFrame({
            "source": ["a", "b", "c", "d", "a"],
            "target": ["b", "c", "d", "a", "c"],
            "weight": [10, 0.5, 1, 3, 13]
        })

    Heterogeneous graphs, with multiple node or edge types, can be created by passing multiple
    DataFrames in a dictionary. The dictionary keys are the names/identifiers for the type. For
    example, if the graph above has node ``a`` of type ``foo``, and the rest as type ``bar``, the
    construction might look like::

        foo_nodes = pd.DataFrame({"x": [-1]}, index=["a"])
        bar_nodes = pd.DataFrame(
            {"y": [0.4, 0.1, 0.9], "z": [100, 200, 300]}, index=["b", "c", "d"]
        )

        StellarGraph({"foo": foo_nodes, "bar": bar_nodes}, edges)

    Notice the ``foo`` node has one feature ``x``, while the ``bar`` nodes have 2 features ``y`` and
    ``z``. A heterogeneous graph can have different features for each type.

    Edges of different types work in the same way. example instance, if edges have different types based
    on their orientation::

        horizontal_edges = pd.DataFrame(
            {"source": ["a", "c"], "target": ["b", "d"]}, index=[0, 2]
        )
        vertical_edges = pd.DataFrame(
            {"source": ["b", "d"], "target": ["c", "a"]}, index=[1, 3]
        )
        diagonal_edges = pd.DataFrame({"source": ["a"], "target": ["c"]}, index=[4])

        StellarGraph(nodes, {"h": horizontal_edges, "v": vertical_edges, "d": diagonal_edges})

    A dictionary can be passed for both arguments::

        StellarGraph(
            {"foo": foo_nodes, "bar": bar_nodes},
            {"h": horizontal_edges, "v": vertical_edges, "d": diagonal_edges}
        )

    .. note::

        The IDs of nodes must be unique across all types: for example, it is an error to have a node
        0 of type ``a``, and a node 0 of type ``b``. IDs of edges must also be unique across all
        types.

    .. seealso:: :meth:`from_networkx` for construction from a NetworkX graph.

    Args:
        nodes (DataFrame or dict of hashable to Pandas DataFrame, optional):
            Features for every node in the graph. Any columns in the DataFrame are taken as numeric
            node features of type ``dtype``. If there is only one type of node, a DataFrame can be
            passed directly, and the type defaults to the ``node_type_default`` parameter. Nodes
            have an ID taken from the index of the dataframe, and they have to be unique across all
            types.  For nodes with no features, an appropriate DataFrame can be created with
            ``pandas.DataFrame([], index=node_ids)``, where ``node_ids`` is a list of the node
            IDs. If this is not passed, the nodes will be inferred from ``edges`` with no features
            for each node.

        edges (DataFrame or dict of hashable to Pandas DataFrame, optional):
            An edge list for each type of edges as a Pandas DataFrame containing a source, target
            and (optionally) weight column (the names of each are taken from the ``source_column``,
            ``target_column`` and ``edge_weight_column`` parameters). If there is only one type of
            edges, a DataFrame can be passed directly, and the type defaults to the
            ``edge_type_default`` parameter. Edges have an ID taken from the index of the dataframe,
            and they have to be unique across all types.

        is_directed (bool, optional):
            If True, the data represents a directed multigraph, otherwise an undirected multigraph.

        source_column (str, optional):
            The name of the column to use as the source node of edges in the ``edges`` edge list
            argument.

        target_column (str, optional):
            The name of the column to use as the target node of edges in the ``edges`` edge list
            argument.

        edge_weight_column (str, optional):
            The name of the column in each of the ``edges`` DataFrames to use as the weight of
            edges. If the column does not exist in any of them, it is defaulted to ``1``.

        node_type_default (str, optional):
            The default node type to use, if ``nodes`` is passed as a DataFrame (not a ``dict``).

        edge_type_default (str, optional):
            The default edge type to use, if ``edges`` is passed as a DataFrame (not a ``dict``).

        dtype (numpy data-type, optional):
            The numpy data-type to use for the features extracted from each of the ``nodes`` DataFrames.

        graph:
            Deprecated, use :meth:`from_networkx`.
        node_type_name:
            Deprecated, use :meth:`from_networkx`.
        edge_type_name:
            Deprecated, use :meth:`from_networkx`.
        node_features:
            Deprecated, use :meth:`from_networkx`.
    """

    def __init__(
        self,
        nodes=None,
        edges=None,
        *,
        is_directed=False,
        source_column=globalvar.SOURCE,
        target_column=globalvar.TARGET,
        edge_weight_column=globalvar.WEIGHT,
        node_type_default=globalvar.NODE_TYPE_DEFAULT,
        edge_type_default=globalvar.EDGE_TYPE_DEFAULT,
        dtype="float32",
        # legacy arguments:
        graph=None,
        node_type_name=globalvar.TYPE_ATTR_NAME,
        edge_type_name=globalvar.TYPE_ATTR_NAME,
        node_features=None,
    ):
        import networkx

        if isinstance(nodes, networkx.Graph):
            # `StellarGraph(nx_graph)` -> `graph`
            graph = nodes
            nodes = None
            if edges is not None:
                raise ValueError(
                    "edges: expected no value when using legacy NetworkX constructor, found: {edges!r}"
                )

        # legacy NetworkX construction
        if graph is not None:
            # FIXME(#717): this should have a deprecation warning, once the tests and examples have
            # stopped using it
            if nodes is not None or edges is not None:
                raise ValueError(
                    "graph: expected no value when using 'nodes' and 'edges' parameters, found: {graph!r}"
                )

            warnings.warn(
                "Constructing a StellarGraph directly from a NetworkX graph has been replaced by the `StellarGraph.from_networkx` function",
                DeprecationWarning,
                stacklevel=2,
            )

            nodes, edges = convert.from_networkx(
                graph,
                node_type_attr=node_type_name,
                edge_type_attr=edge_type_name,
                node_type_default=node_type_default,
                edge_type_default=edge_type_default,
                edge_weight_attr=edge_weight_column,
                node_features=node_features,
                dtype=dtype,
            )

        if edges is None:
            edges = {}

        self._is_directed = is_directed
        self._edges = convert.convert_edges(
            edges,
            name="edges",
            default_type=edge_type_default,
            source_column=source_column,
            target_column=target_column,
            weight_column=edge_weight_column,
        )

        nodes_from_edges = pd.unique(
            np.concatenate([self._edges.targets, self._edges.sources])
        )

        if nodes is None:
            nodes_after_inference = pd.DataFrame([], index=nodes_from_edges)
        else:
            nodes_after_inference = nodes

        self._nodes = convert.convert_nodes(
            nodes_after_inference,
            name="nodes",
            default_type=node_type_default,
            dtype=dtype,
        )

        if nodes is not None:
            # check for dangling edges: make sure the explicitly-specified nodes parameter includes every
            # node mentioned in the edges
            try:
                self._nodes.ids.to_iloc(
                    nodes_from_edges, smaller_type=False, strict=True,
                )
            except KeyError as e:
                missing_values = e.args[0]
                if not is_real_iterable(missing_values):
                    missing_values = [missing_values]
                missing_values = pd.unique(missing_values)

                raise ValueError(
                    f"edges: expected all source and target node IDs to be contained in `nodes`, "
                    f"found some missing: {comma_sep(missing_values)}"
                )

    @staticmethod
    def from_networkx(
        graph,
        *,
        edge_weight_attr="weight",
        node_type_attr=globalvar.TYPE_ATTR_NAME,
        edge_type_attr=globalvar.TYPE_ATTR_NAME,
        node_type_default=globalvar.NODE_TYPE_DEFAULT,
        edge_type_default=globalvar.EDGE_TYPE_DEFAULT,
        node_features=None,
        dtype="float32",
    ):
        """
        Construct a ``StellarGraph`` object from a NetworkX graph::

            Gs = StellarGraph.from_networkx(nx_graph)

        To create a StellarGraph object with node features, supply the features
        as a numeric feature vector for each node.

        To take the feature vectors from a node attribute in the original NetworkX
        graph, supply the attribute name to the ``node_features`` argument::

            Gs = StellarGraph.from_networkx(nx_graph, node_features="feature")


        where the nx_graph contains nodes that have a "feature" attribute containing
        the feature vector for the node. All nodes of the same type must have
        the same size feature vectors.

        Alternatively, supply the node features as Pandas DataFrame objects with
        the index of the DataFrame set to the node IDs. For graphs with a single node
        type, you can supply the DataFrame object directly to StellarGraph::

            node_data = pd.DataFrame(
                [feature_vector_1, feature_vector_2, ..],
                index=[node_id_1, node_id_2, ...])
            Gs = StellarGraph.from_networkx(nx_graph, node_features=node_data)

        For graphs with multiple node types, provide the node features as Pandas
        DataFrames for each type separately, as a dictionary by node type.
        This allows node features to have different sizes for each node type::

            node_data = {
                node_type_1: pd.DataFrame(...),
                node_type_2: pd.DataFrame(...),
            }
            Gs = StellarGraph.from_networkx(nx_graph, node_features=node_data)

        The dictionary only needs to include node types with features. If a node type isn't
        mentioned in the dictionary (for example, if `nx_graph` above has a 3rd node type), each
        node of that type will have a feature vector of length zero.

        You can also supply the node feature vectors as an iterator of `node_id`
        and feature vector pairs, for graphs with single and multiple node types::

            node_data = zip([node_id_1, node_id_2, ...],
                [feature_vector_1, feature_vector_2, ..])
            Gs = StellarGraph.from_networkx(nx_graph, node_features=node_data)


        Args:
            graph: The NetworkX graph instance.
            node_type_attr (str, optional):
                This is the name for the node types that StellarGraph uses
                when processing heterogeneous graphs. StellarGraph will
                look for this attribute in the nodes of the graph to determine
                their type.

            node_type_default (str, optional):
                This is the default node type to use for nodes that do not have
                an explicit type.

            edge_type_attr (str, optional):
                This is the name for the edge types that StellarGraph uses
                when processing heterogeneous graphs. StellarGraph will
                look for this attribute in the edges of the graph to determine
                their type.

            edge_type_default (str, optional):
                This is the default edge type to use for edges that do not have
                an explicit type.

            node_features (str, dict, list or DataFrame optional):
                This tells StellarGraph where to find the node feature information
                required by some graph models. These are expected to be
                a numeric feature vector for each node in the graph.

            edge_weight_attr (str, optional):
                The name of the attribute to use as the weight of edges.

        Returns:
            A ``StellarGraph`` (if ``graph`` is undirected) or ``StellarDiGraph`` (if ``graph`` is
            directed) instance representing the data in ``graph`` and ``node_features``.
        """
        nodes, edges = convert.from_networkx(
            graph,
            node_type_attr=node_type_attr,
            edge_type_attr=edge_type_attr,
            node_type_default=node_type_default,
            edge_type_default=edge_type_default,
            edge_weight_attr=edge_weight_attr,
            node_features=node_features,
            dtype=dtype,
        )

        cls = StellarDiGraph if graph.is_directed() else StellarGraph
        return cls(
            nodes=nodes, edges=edges, edge_weight_column=edge_weight_attr, dtype=dtype
        )

    # customise how a missing attribute is handled to give better error messages for the NetworkX
    # -> no NetworkX transition.
    def __getattr__(self, item):
        import networkx

        try:
            # do the normal access, in case the attribute actually exists, and to get the native
            # python wording of the error
            return super().__getattribute__(item)
        except AttributeError as e:
            if hasattr(networkx.MultiDiGraph, item):
                # a networkx class has this as an attribute, so let's assume that it's old code
                # from before the conversion and replace (the `from None`) the default exception
                # with one with a more specific message that guides the user to the fix
                type_name = type(self).__name__
                raise AttributeError(
                    f"{e.args[0]}. The '{type_name}' type no longer inherits from NetworkX types: use a new StellarGraph method, or, if that is not possible, the `.to_networkx()` conversion function."
                ) from None

            # doesn't look like a NetworkX method so use the default error
            raise

    def is_directed(self) -> bool:
        """
        Indicates whether the graph is directed (True) or undirected (False).

        Returns:
             bool: The graph directedness status.
        """
        return self._is_directed

    def number_of_nodes(self) -> int:
        """
        Obtains the number of nodes in the graph.

        Returns:
             int: The number of nodes.
        """
        return len(self._nodes)

    def number_of_edges(self) -> int:
        """
        Obtains the number of edges in the graph.

        Returns:
             int: The number of edges.
        """
        return len(self._edges)

    def nodes(self, node_type=None) -> Iterable[Any]:
        """
        Obtains the collection of nodes in the graph.

        Args:
            node_type (hashable, optional): a type of nodes that exist in the graph

        Returns:
            All the nodes in the graph if ``node_type`` is ``None``, otherwise all the nodes in the
            graph of type ``node_type``.
        """
        if node_type is None:
            return self._nodes.ids.pandas_index

        ilocs = self._nodes.type_range(node_type)
        return self._nodes.ids.from_iloc(ilocs)

    def edges(
        self, include_edge_type=False, include_edge_weight=False
    ) -> Iterable[Any]:
        """
        Obtains the collection of edges in the graph.

        Args:
            include_edge_type (bool): A flag that indicates whether to return edge types
            of format (node 1, node 2, edge type) or edge pairs of format (node 1, node 2).
            include_edge_weight (bool): A flag that indicates whether to return edge weights.
            Weights are returned in a separate list.

        Returns:
            The graph edges. If edge weights are included then a tuple of (edges, weights)
        """
        # FIXME: these would be better returned as the 2 or 3 arrays directly, rather than tuple-ing
        # (the same applies to all other instances of zip in this file)
        if include_edge_type:
            edges = list(
                zip(
                    self._edges.sources,
                    self._edges.targets,
                    self._edges.type_of_iloc(slice(None)),
                )
            )
        else:
            edges = list(zip(self._edges.sources, self._edges.targets))

        if include_edge_weight:
            return edges, self._edges.weights

        return edges

    def has_node(self, node: Any) -> bool:
        """
        Indicates whether or not the graph contains the specified node.

        Args:
            node (any): The node.

        Returns:
             bool: A value of True (cf False) if the node is
             (cf is not) in the graph.
        """
        return node in self._nodes

    def _transform_edges(
        self, other_node_id, ilocs, include_edge_weight, filter_edge_types
    ):
        if include_edge_weight:
            weights = self._edges.weights[ilocs]
        else:
            weights = None

        if filter_edge_types is not None:
            filter_edge_type_ilocs = self._edges.types.to_iloc(filter_edge_types)
            edge_type_ilocs = self._edges.type_ilocs[ilocs]
            correct_type = np.isin(edge_type_ilocs, filter_edge_type_ilocs)

            other_node_id = other_node_id[correct_type]
            if weights is not None:
                weights = weights[correct_type]

        # FIXME(#718): it would be better to return these as ndarrays, instead of (zipped) lists
        if weights is not None:
            return [
                NeighbourWithWeight(node, weight)
                for node, weight in zip(other_node_id, weights)
            ]

        return list(other_node_id)

    def neighbors(
        self, node: Any, include_edge_weight=False, edge_types=None
    ) -> Iterable[Any]:
        """
        Obtains the collection of neighbouring nodes connected
        to the given node.

        Args:
            node (any): The node in question.
            include_edge_weight (bool, default False): If True, each neighbour in the
                output is a named tuple with fields `node` (the node ID) and `weight` (the edge weight)
            edge_types (list of hashable, optional): If provided, only traverse the graph
                via the provided edge types when collecting neighbours.

        Returns:
            iterable: The neighbouring nodes.
        """
        ilocs = self._edges.edge_ilocs(node, ins=True, outs=True)
        source = self._edges.sources[ilocs]
        target = self._edges.targets[ilocs]
        other_node_id = np.where(source == node, target, source)
        return self._transform_edges(
            other_node_id, ilocs, include_edge_weight, edge_types
        )

    def in_nodes(
        self, node: Any, include_edge_weight=False, edge_types=None
    ) -> Iterable[Any]:
        """
        Obtains the collection of neighbouring nodes with edges
        directed to the given node. For an undirected graph,
        neighbours are treated as both in-nodes and out-nodes.

        Args:
            node (any): The node in question.
            include_edge_weight (bool, default False): If True, each neighbour in the
                output is a named tuple with fields `node` (the node ID) and `weight` (the edge weight)
            edge_types (list of hashable, optional): If provided, only traverse the graph
                via the provided edge types when collecting neighbours.

        Returns:
            iterable: The neighbouring in-nodes.
        """
        if not self.is_directed():
            # all edges are both incoming and outgoing for undirected graphs
            return self.neighbors(
                node, include_edge_weight=include_edge_weight, edge_types=edge_types
            )

        ilocs = self._edges.edge_ilocs(node, ins=True, outs=False)
        source = self._edges.sources[ilocs]
        return self._transform_edges(source, ilocs, include_edge_weight, edge_types)

    def out_nodes(
        self, node: Any, include_edge_weight=False, edge_types=None
    ) -> Iterable[Any]:
        """
        Obtains the collection of neighbouring nodes with edges
        directed from the given node. For an undirected graph,
        neighbours are treated as both in-nodes and out-nodes.

        Args:
            node (any): The node in question.
            include_edge_weight (bool, default False): If True, each neighbour in the
                output is a named tuple with fields `node` (the node ID) and `weight` (the edge weight)
            edge_types (list of hashable, optional): If provided, only traverse the graph
                via the provided edge types when collecting neighbours.

        Returns:
            iterable: The neighbouring out-nodes.
        """
        if not self.is_directed():
            # all edges are both incoming and outgoing for undirected graphs
            return self.neighbors(
                node, include_edge_weight=include_edge_weight, edge_types=edge_types
            )

        ilocs = self._edges.edge_ilocs(node, ins=False, outs=True)
        target = self._edges.targets[ilocs]
        return self._transform_edges(target, ilocs, include_edge_weight, edge_types)

    def nodes_of_type(self, node_type=None):
        """
        Get the nodes of the graph with the specified node types.

        Args:
            node_type (hashable): a type of nodes that exist in the graph (this must be passed,
                omitting it or passing ``None`` is deprecated)

        Returns:
            A list of node IDs with type node_type
        """
        warnings.warn(
            "'nodes_of_type' is deprecated and will be removed; use the 'nodes(type=...)' method instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return list(self.nodes(node_type=node_type))

    def node_type(self, node):
        """
        Get the type of the node

        Args:
            node: Node ID

        Returns:
            Node type
        """
        nodes = [node]
        node_ilocs = self._nodes.ids.to_iloc(nodes, strict=True)
        type_sequence = self._nodes.type_of_iloc(node_ilocs)

        assert len(type_sequence) == 1
        return type_sequence[0]

    @property
    def node_types(self):
        """
        Get a list of all node types in the graph.

        Returns:
            set of types
        """
        return set(self._nodes.types.pandas_index)

    def node_feature_sizes(self, node_types=None):
        """
        Get the feature sizes for the specified node types.

        Args:
            node_types (list, optional): A list of node types. If None all current node types
                will be used.

        Returns:
            A dictionary of node type and integer feature size.
        """
        all_sizes = self._nodes.feature_info()

        if node_types is None:
            node_types = all_sizes.keys()

        return {type_name: all_sizes[type_name][0] for type_name in node_types}

    def check_graph_for_ml(self, features=True):
        """
        Checks if all properties required for machine learning training/inference are set up.
        An error will be raised if the graph is not correctly setup.
        """
        if all(size == 0 for _, size in self.node_feature_sizes().items()):
            raise RuntimeError(
                "This StellarGraph has no numeric feature attributes for nodes"
                "Node features are required for machine learning"
            )

        # TODO: check the schema

        # TODO: check the feature node_ids against the graph node ids?

    def node_features(self, nodes, node_type=None) -> np.ndarray:
        """
        Get the numeric feature vectors for the specified node or nodes.
        If the node type is not specified the node types will be found
        for all nodes. It is therefore important to supply the ``node_type``
        for this method to be fast.

        Args:
            nodes (list or hashable): Node ID or list of node IDs
            node_type (hashable): the type of the nodes.

        Returns:
            Numpy array containing the node features for the requested nodes.
        """
        feature_tensor = self.node_features_tensors(nodes, node_type)

        if tf.executing_eagerly():
            return feature_tensor.numpy()

        with tf.compat.v1.Session() as sess:
            try:
                return sess.run(feature_tensor)
            except InvalidArgumentError:
                raise ValueError("unknown IDs")

    def node_features_tensors(self, nodes, node_type=None) -> tf.Tensor:
        """
        Get the numeric feature vectors for the specified node or nodes.
        If the node type is not specified the node types will be found
        for all nodes. It is therefore important to supply the ``node_type``
        for this method to be fast.

        Args:
            nodes (list or hashable): Node ID or list of node IDs
            node_type (hashable): the type of the nodes.

        Returns:
            tensorflow Tensor containing the node features for the requested nodes.
        """
        nodes = np.asarray(nodes)

        node_ilocs = self._nodes.ids.to_iloc(nodes)
        valid = self._nodes.ids.is_valid(node_ilocs)
        all_valid = valid.all()
        valid_ilocs = node_ilocs if all_valid else node_ilocs[valid]

        if node_type is None:
            # infer the type based on the valid nodes
            types = np.unique(self._nodes.type_of_iloc(valid_ilocs))

            if len(types) == 0:
                raise ValueError(
                    "must have at least one node for inference, if `node_type` is not specified"
                )
            if len(types) > 1:
                raise ValueError("all nodes must have the same type")

            node_type = types[0]

        if all_valid:
            return self._nodes.features(node_type, valid_ilocs)

        # If there's some invalid values, they get replaced by zeros; this is designed to allow
        # models that build fixed-size structures (e.g. GraphSAGE) based on neighbours to fill out
        # missing neighbours with zeros automatically, using None as a sentinel.

        # FIXME: None as a sentinel forces nodes to have dtype=object even with integer IDs, could
        # instead use an impossible integer (e.g. 2**64 - 1)

        # everything that's not the sentinel should be valid
        non_nones = nodes != None
        self._nodes.ids.require_valid(nodes[non_nones], node_ilocs[non_nones])

        with tf.device("/CPU:0"):
            sampled = self._nodes.features(node_type, valid_ilocs)

            indices = tf.cast(tf.where(valid), tf.int32)
            shape = (len(nodes), sampled.shape[1])
            features = tf.scatter_nd(indices, sampled, shape)

        return features

    ##################################################################
    # Computationally intensive methods:

    def _edge_type_iloc_triples(self, selector=slice(None), stacked=False):
        source_ilocs = self._nodes.ids.to_iloc(self._edges.sources[selector])
        source_type_ilocs = self._nodes.type_ilocs[source_ilocs]

        rel_type_ilocs = self._edges.type_ilocs[selector]

        target_ilocs = self._nodes.ids.to_iloc(self._edges.targets[selector])
        target_type_ilocs = self._nodes.type_ilocs[target_ilocs]

        all_ilocs = source_type_ilocs, rel_type_ilocs, target_type_ilocs
        if stacked:
            return np.stack(all_ilocs, axis=-1)

        return all_ilocs

    def _edge_type_triples(self, selector=slice(None)):
        src_ilocs, rel_ilocs, tgt_ilocs = self._edge_type_iloc_triples(
            selector, stacked=False
        )

        return (
            self._nodes.types.from_iloc(src_ilocs),
            self._edges.types.from_iloc(rel_ilocs),
            self._nodes.types.from_iloc(tgt_ilocs),
        )

    def _unique_type_triples(self, *, return_counts, selector=slice(None)):
        all_type_ilocs = self._edge_type_iloc_triples(selector, stacked=True)

        if len(all_type_ilocs) == 0:
            # FIXME(https://github.com/numpy/numpy/issues/15559): if there's no edges, np.unique is
            # being called on a shape=(0, 3) ndarray, and hits "ValueError: cannot reshape array of
            # size 0 into shape (0,newaxis)", so we manually reproduce what would be returned
            if return_counts:
                ret = None, [], []
            else:
                ret = None, []
        else:
            ret = np.unique(
                all_type_ilocs, axis=0, return_index=True, return_counts=return_counts
            )

        edge_ilocs = ret[1]
        # we've now got the indices for an edge with each triple, along with the counts of them, so
        # we can query to get the actual edge types (this is, at the time of writing, easier than
        # getting the actual type for each type iloc in the triples)
        unique_ets = self._edge_type_triples(edge_ilocs)

        if return_counts:
            return zip(*unique_ets, ret[2])

        return zip(*unique_ets)

    def info(self, show_attributes=None, sample=None, truncate=20):
        """
        Return an information string summarizing information on the current graph.
        This includes node and edge type information and their attributes.

        Note: This requires processing all nodes and edges and could take a long
        time for a large graph.

        Args:
            show_attributes: Deprecated, unused.
            sample: Deprecated, unused.
            truncate (int, optional): If an integer, show only the ``truncate`` most common node and
                edge type triples; if ``None``, list each one individually.
        Returns:
            An information string.
        """
        if show_attributes is not None:
            warnings.warn(
                "'show_attributes' is no longer used, remove it from the 'info()' call",
                DeprecationWarning,
                stacklevel=2,
            )

        if sample is not None:
            warnings.warn(
                "'sample' is no longer used, remove it from the 'info()' call",
                DeprecationWarning,
                stacklevel=2,
            )

        # always truncate the edge types listed for each node type, since they're redundant with the
        # individual listing of edge types, and make for a single very long line
        truncate_edge_types_per_node = 5
        if truncate is not None:
            truncate_edge_types_per_node = min(truncate_edge_types_per_node, truncate)

        # Numpy processing is much faster than NetworkX processing, so we don't bother sampling.
        gs = self.create_graph_schema()

        feature_info = self._nodes.feature_info()

        def str_edge_type(et):
            n1, rel, n2 = et
            return f"{n1}-{rel}->{n2}"

        def str_node_type(count, nt):
            feature_size, feature_dtype = feature_info[nt]
            if feature_size > 0:
                feature_text = f"{feature_dtype.name} vector, length {feature_size}"
            else:
                feature_text = "none"

            edges = gs.schema[nt]
            if edges:
                edge_types = comma_sep(
                    [str_edge_type(et) for et in gs.schema[nt]],
                    limit=truncate_edge_types_per_node,
                    stringify=str,
                )
            else:
                edge_types = "none"
            return f"{nt}: [{count}]\n    Features: {feature_text}\n    Edge types: {edge_types}"

        # sort the node types in decreasing order of frequency
        node_types = sorted(
            ((len(self.nodes(node_type=nt)), nt) for nt in gs.node_types), reverse=True
        )
        nodes = separated(
            [str_node_type(count, nt) for count, nt in node_types],
            limit=truncate,
            stringify=str,
            sep="\n  ",
        )

        # FIXME: it would be better for the schema to just include the counts directly
        unique_ets = self._unique_type_triples(return_counts=True)
        edge_types = sorted(
            (
                (count, EdgeType(src_ty, rel_ty, tgt_ty))
                for src_ty, rel_ty, tgt_ty, count in unique_ets
            ),
            reverse=True,
        )

        edges = separated(
            [f"{str_edge_type(et)}: [{count}]" for count, et in edge_types],
            limit=truncate,
            stringify=str,
            sep="\n    ",
        )

        directed_str = "Directed" if self.is_directed() else "Undirected"
        lines = [
            f"{type(self).__name__}: {directed_str} multigraph",
            f" Nodes: {self.number_of_nodes()}, Edges: {self.number_of_edges()}",
            "",
            " Node types:",
        ]
        if nodes:
            lines.append("  " + nodes)

        lines.append("")
        lines.append(" Edge types:")

        if edges:
            lines.append("    " + edges)

        return "\n".join(lines)

    def create_graph_schema(self, nodes=None):
        """
        Create graph schema from the current graph.

        Arguments:
            nodes (list): A list of node IDs to use to build schema. This must
                represent all node types and all edge types in the graph.
                If not specified, all nodes and edges in the graph are used.

        Returns:
            GraphSchema object.
        """

        graph_schema = {nt: set() for nt in self.node_types}
        edge_types = set()

        if nodes is None:
            selector = slice(None)
        else:
            selector = np.isin(self._edges.sources, nodes) & np.isin(
                self._edges.targets, nodes
            )

        for n1, rel, n2 in self._unique_type_triples(
            selector=selector, return_counts=False
        ):
            edge_type_tri = EdgeType(n1, rel, n2)
            edge_types.add(edge_type_tri)
            graph_schema[n1].add(edge_type_tri)

            if not self.is_directed():
                edge_type_tri = EdgeType(n2, rel, n1)
                edge_types.add(edge_type_tri)
                graph_schema[n2].add(edge_type_tri)

        # Create ordered list of edge_types
        edge_types = sorted(edge_types)

        # Create keys for node and edge types
        schema = {
            node_label: sorted(node_data)
            for node_label, node_data in graph_schema.items()
        }

        return GraphSchema(
            self.is_directed(), sorted(self.node_types), edge_types, schema
        )

    def node_degrees(self) -> Mapping[Any, int]:
        """
        Obtains a map from node to node degree.

        Returns:
            The degree of each node.
        """
        return self._edges.degrees()

    def to_adjacency_matrix(self, nodes: Optional[Iterable] = None, weighted=False):
        """
        Obtains a SciPy sparse adjacency matrix of edge weights.

        By default (``weighted=False``), each element of the matrix contains the number
        of edges between the two vertices (only 0 or 1 in a graph without multi-edges).

        Args:
            nodes (iterable): The optional collection of nodes
                comprising the subgraph. If specified, then the
                adjacency matrix is computed for the subgraph;
                otherwise, it is computed for the full graph.
            weighted (bool): If true, use the edge weight column from the graph instead
                of edge counts (weights from multi-edges are summed).

        Returns:
             The weighted adjacency matrix.
        """
        if nodes is None:
            index = self._nodes._id_index
            selector = slice(None)
        else:
            nodes = list(nodes)
            index = ExternalIdIndex(nodes)
            selector = np.isin(self._edges.sources, nodes) & np.isin(
                self._edges.targets, nodes
            )

        # these indices are computed relative to the index above. If `nodes` is None, they'll be the
        # overall ilocs (for the original graph), otherwise they'll be the indices of the `nodes`
        # list.
        src_idx = index.to_iloc(self._edges.sources[selector])
        tgt_idx = index.to_iloc(self._edges.targets[selector])
        if weighted:
            weights = self._edges.weights[selector]
        else:
            weights = np.ones(src_idx.shape, dtype=self._edges.weights.dtype)

        n = len(index)

        adj = sps.csr_matrix((weights, (src_idx, tgt_idx)), shape=(n, n))
        if not self.is_directed():
            # in an undirected graph, the adjacency matrix should be symmetric: which means counting
            # weights from either "incoming" or "outgoing" edges, but not double-counting self loops
            backward = adj.transpose(copy=True)
            # this is setdiag(0), but faster, since it doesn't change the sparsity structure of the
            # matrix (https://github.com/scipy/scipy/issues/11600)
            (nonzero,) = backward.diagonal().nonzero()
            backward[nonzero, nonzero] = 0

            adj += backward

        # this is a multigraph, let's eliminate any duplicate entries
        adj.sum_duplicates()
        return adj

    def subgraph(self, nodes):
        """
        Compute the node-induced subgraph implied by ``nodes``.

        Args:
            nodes (iterable): The nodes in the subgraph.

        Returns:
            A :class:`StellarGraph` or :class:`StellarDiGraph` instance containing only the nodes in
            ``nodes``, and any edges between them in ``self``. It contains the same node & edge
            types, node features and edge weights as in ``self``.
        """

        node_ilocs = self._nodes.ids.to_iloc(nodes, strict=True)
        node_types = self._nodes.type_of_iloc(node_ilocs)
        node_type_to_ids = pd.Series(nodes, index=node_types).groupby(level=0)

        node_frames = {
            type_name: pd.DataFrame(
                self.node_features(type_node_ids, type_name), index=type_node_ids,
            )
            for type_name, type_node_ids in node_type_to_ids
        }

        # FIXME(#985): this is O(edges in graph) but could potentially be optimised to O(edges in
        # graph incident to `nodes`), which could be much fewer if `nodes` is small
        edge_ilocs = np.where(
            np.isin(self._edges.sources, nodes) & np.isin(self._edges.targets, nodes)
        )
        edge_frame = pd.DataFrame(
            {
                "id": self._edges.ids.from_iloc(edge_ilocs),
                globalvar.SOURCE: self._edges.sources[edge_ilocs],
                globalvar.TARGET: self._edges.targets[edge_ilocs],
                globalvar.WEIGHT: self._edges.weights[edge_ilocs],
            },
            index=self._edges.type_of_iloc(edge_ilocs),
        )
        edge_frames = {
            type_name: df.set_index("id")
            for type_name, df in edge_frame.groupby(level=0)
        }

        cls = StellarDiGraph if self.is_directed() else StellarGraph
        return cls(node_frames, edge_frames)

    def connected_components(self):
        """
        Compute the connected components in this graph, ordered by size.

        The nodes in the largest component can be computed with ``nodes =
        next(graph.connected_components())``. The node IDs returned by this method can be used to
        compute the corresponding subgraph with ``graph.subgraph(nodes)``.

        For directed graphs, this computes the weakly connected components. This effectively
        treating each edge as undirected.

        Returns:
            An iterator over sets of node IDs in each connected component, from the largest (most nodes)
            to smallest (fewest nodes).
        """

        adj = self.to_adjacency_matrix()
        count, cc_labels = sps.csgraph.connected_components(adj, directed=False)
        cc_sizes = np.bincount(cc_labels, minlength=count)
        cc_by_size = np.argsort(cc_sizes)[::-1]

        return (
            self._nodes.ids.from_iloc(cc_labels == cc_label) for cc_label in cc_by_size
        )

    def to_networkx(
        self,
        node_type_attr=globalvar.TYPE_ATTR_NAME,
        edge_type_attr=globalvar.TYPE_ATTR_NAME,
        edge_weight_attr=globalvar.WEIGHT,
        feature_attr=globalvar.FEATURE_ATTR_NAME,
        node_type_name=None,
        edge_type_name=None,
        edge_weight_label=None,
        feature_name=None,
    ):
        """
        Create a NetworkX MultiGraph or MultiDiGraph instance representing this graph.

        Args:
            node_type_attr (str): the name of the attribute to use to store a node's type (or label).

            edge_type_attr (str): the name of the attribute to use to store a edge's type (or label).

            edge_weight_attr (str): the name of the attribute to use to store a edge's weight.

            feature_attr (str, optional): the name of the attribute to use to store a node's feature
                vector; if ``None``, feature vectors are not stored within each node.

            node_type_name (str): Deprecated, use ``node_type_attr``.
            edge_type_name (str): Deprecated, use ``edge_type_attr``.
            edge_weight_label (str): Deprecated, use ``edge_weight_attr``.
            feature_name (str, optional): Deprecated, use ``feature_attr``.

        Returns:
             An instance of `networkx.MultiDiGraph` (if directed) or `networkx.MultiGraph` (if
             undirected) containing all the nodes & edges and their types & features in this graph.
        """
        import networkx

        if node_type_name is not None:
            warnings.warn(
                "the 'node_type_name' parameter has been replaced by 'node_type_attr'",
                DeprecationWarning,
                stacklevel=2,
            )
            node_type_attr = node_type_name

        if edge_type_name is not None:
            warnings.warn(
                "the 'edge_type_name' parameter has been replaced by 'edge_type_attr'",
                DeprecationWarning,
                stacklevel=2,
            )
            edge_type_attr = edge_type_name

        if edge_weight_label is not None:
            warnings.warn(
                "the 'edge_weight_label' parameter has been replaced by 'edge_weight_attr'",
                DeprecationWarning,
                stacklevel=2,
            )
            edge_weight_attr = edge_weight_label

        if feature_name is not None:
            warnings.warn(
                "the 'feature_name' parameter has been replaced by 'feature_attr'",
                DeprecationWarning,
                stacklevel=2,
            )
            feature_attr = feature_name

        if self.is_directed():
            graph = networkx.MultiDiGraph()
        else:
            graph = networkx.MultiGraph()

        for ty in self.node_types:
            node_ids = self.nodes(node_type=ty)
            ty_dict = {node_type_attr: ty}

            if feature_attr is not None:
                features = self.node_features(node_ids, node_type=ty)

                for node_id, node_features in zip(node_ids, features):
                    graph.add_node(
                        node_id, **ty_dict, **{feature_attr: node_features},
                    )
            else:
                graph.add_nodes_from(node_ids, **ty_dict)

        iterator = zip(
            self._edges.sources,
            self._edges.targets,
            self._edges.type_of_iloc(slice(None)),
            self._edges.weights,
        )
        graph.add_edges_from(
            (src, dst, {edge_type_attr: type_, edge_weight_attr: weight})
            for src, dst, type_, weight in iterator
        )

        return graph

    # FIXME: Experimental/special-case methods that need to be considered more; the underscores
    # denote "package private", not fully private, and so are ok to use in the rest of stellargraph
    def _get_index_for_nodes(self, nodes, node_type=None):
        """
        Get the indices for the specified node or nodes.
        If the node type is not specified the node types will be found
        for all nodes. It is therefore important to supply the ``node_type``
        for this method to be fast.

        Args:
            n: (list or hashable) Node ID or list of node IDs
            node_type: (hashable) the type of the nodes.

        Returns:
            Numpy array containing the indices for the requested nodes.
        """
        return self._nodes._id_index.to_iloc(nodes, strict=True)

    def _adjacency_types(self, graph_schema: GraphSchema):
        """
        Obtains the edges in the form of the typed mapping:

            {edge_type_triple: {source_node: [target_node, ...]}}

        Args:
            graph_schema: The graph schema.
        Returns:
             The edge types mapping.
        """
        source_types, rel_types, target_types = self._edge_type_triples(slice(None))

        triples = defaultdict(lambda: defaultdict(lambda: []))

        iterator = zip(
            source_types,
            rel_types,
            target_types,
            self._edges.sources,
            self._edges.targets,
        )
        for src_type, rel_type, tgt_type, src, tgt in iterator:
            triple = EdgeType(src_type, rel_type, tgt_type)
            triples[triple][src].append(tgt)

            if not self.is_directed() and src != tgt:
                other_triple = EdgeType(tgt_type, rel_type, src_type)
                triples[other_triple][tgt].append(src)

        for subdict in triples.values():
            for v in subdict.values():
                # each list should be in order, to ensure sampling methods are deterministic
                v.sort(key=str)

        return triples

    def _edge_weights(self, source_node: Any, target_node: Any) -> List[Any]:
        """
        Obtains the weights of edges between the given pair of nodes.

        Args:
            source_node (any): The source node.
            target_node (any): The target node.

        Returns:
            list: The edge weights.
        """
        # self loops should only be counted once, which means they're effectively always a directed
        # edge at the storage level, unlikely other edges in an undirected graph. This is
        # particularly important with the intersection1d call, where the source_ilocs and
        # target_ilocs will be equal, when source_node == target_node, and thus the intersection
        # will contain all incident edges.
        effectively_directed = self.is_directed() or source_node == target_node
        both_dirs = not effectively_directed

        source_ilocs = self._edges.edge_ilocs(source_node, ins=both_dirs, outs=True)
        target_ilocs = self._edges.edge_ilocs(target_node, ins=True, outs=both_dirs)

        ilocs = np.intersect1d(source_ilocs, target_ilocs, assume_unique=True)

        return [float(x) for x in self._edges.weights[ilocs]]


# A convenience class that merely specifies that edges have direction.
class StellarDiGraph(StellarGraph):
    def __init__(
        self,
        nodes=None,
        edges=None,
        *,
        source_column=globalvar.SOURCE,
        target_column=globalvar.TARGET,
        edge_weight_column=globalvar.WEIGHT,
        node_type_default=globalvar.NODE_TYPE_DEFAULT,
        edge_type_default=globalvar.EDGE_TYPE_DEFAULT,
        dtype="float32",
        # legacy arguments
        graph=None,
        node_type_name=globalvar.TYPE_ATTR_NAME,
        edge_type_name=globalvar.TYPE_ATTR_NAME,
        node_features=None,
    ):
        super().__init__(
            nodes=nodes,
            edges=edges,
            is_directed=True,
            source_column=source_column,
            target_column=target_column,
            edge_weight_column=edge_weight_column,
            node_type_default=node_type_default,
            edge_type_default=edge_type_default,
            dtype=dtype,
            # legacy arguments
            graph=graph,
            node_type_name=node_type_name,
            edge_type_name=edge_type_name,
            node_features=node_features,
        )
