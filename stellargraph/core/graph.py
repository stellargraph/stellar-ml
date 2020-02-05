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

from .. import globalvar
from .schema import GraphSchema, EdgeType
from .experimental import experimental, ExperimentalWarning
from .element_data import NodeData, EdgeData, ExternalIdIndex
from .utils import is_real_iterable
from .validation import comma_sep
from . import convert


NeighbourWithWeight = namedtuple("NeighbourWithWeight", ["node", "weight"])


class StellarGraph:
    """
    StellarGraph class for directed or undirected graph ML models. It stores both
    graph structure and features for machine learning.

    To create a StellarGraph object ready for machine learning, at a
    minimum pass the graph structure to the StellarGraph as a NetworkX
    graph:

    For undirected models::

        Gs = StellarGraph(nx_graph)


    For directed models::

        Gs = StellarDiGraph(nx_graph)


    To create a StellarGraph object with node features, supply the features
    as a numeric feature vector for each node.

    To take the feature vectors from a node attribute in the original NetworkX
    graph, supply the attribute name to the ``node_features`` argument::

        Gs = StellarGraph(nx_graph, node_features="feature")


    where the nx_graph contains nodes that have a "feature" attribute containing
    the feature vector for the node. All nodes of the same type must have
    the same size feature vectors.

    Alternatively, supply the node features as Pandas DataFrame objects with
    the of the DataFrame set to the node IDs. For graphs with a single node
    type, you can supply the DataFrame object directly to StellarGraph::

        node_data = pd.DataFrame(
            [feature_vector_1, feature_vector_2, ..],
            index=[node_id_1, node_id_2, ...])
        Gs = StellarGraph(nx_graph, node_features=node_data)

    For graphs with multiple node types, provide the node features as Pandas
    DataFrames for each type separately, as a dictionary by node type.
    This allows node features to have different sizes for each node type::

        node_data = {
            node_type_1: pd.DataFrame(...),
            node_type_2: pd.DataFrame(...),
        }
        Gs = StellarGraph(nx_graph, node_features=node_data)


    You can also supply the node feature vectors as an iterator of `node_id`
    and feature vector pairs, for graphs with single and multiple node types::

        node_data = zip([node_id_1, node_id_2, ...],
            [feature_vector_1, feature_vector_2, ..])
        Gs = StellarGraph(nx_graph, node_features=node_data)

    .. warning::

        The constructor variant using the ``nodes=..., edges=...`` arguments to create a "new"
        StellarGraph is experimental: the type is insufficiently documented and does not support
        some algorithms.


    Args:
        graph: The NetworkX graph instance.
        node_type_name: str, optional (default=globals.TYPE_ATTR_NAME)
            This is the name for the node types that StellarGraph uses
            when processing heterogeneous graphs. StellarGraph will
            look for this attribute in the nodes of the graph to determine
            their type.

        node_type_default: str, optional (default=globals.NODE_TYPE_DEFAULT)
            This is the default node type to use for nodes that do not have
            an explicit type.

        edge_type_name: str, optional (default=globals.TYPE_ATTR_NAME)
            This is the name for the edge types that StellarGraph uses
            when processing heterogeneous graphs. StellarGraph will
            look for this attribute in the edges of the graph to determine
            their type.

        edge_type_default: str, optional (default=globals.EDGE_TYPE_DEFAULT)
            This is the default edge type to use for edges that do not have
            an explicit type.

        node_features: str, dict, list or DataFrame optional (default=None)
            This tells StellarGraph where to find the node feature information
            required by some graph models. These are expected to be
            a numeric feature vector for each node in the graph.

        nodes: DataFrame or dict of hashable to DataFrame
            Features for every node in the graph. Any columns in the dataframe are taken as numeric
            node features of type ``dtype``. If there is only one type of node, a DataFrame can be
            passed directly, and the type defaults to the ``node_type_default`` parameter. Nodes
            have an ID taken from the index of the dataframe, and they have to be unique across all
            types.  For nodes with no features, an appropriate DataFrame can be created with
            ``pandas.DataFrame([], index=node_ids)``, where ``node_ids`` is a list of the node
            IDs. This must be used with the ``edges`` argument. This uses the same basic structure as the
            ``node_features`` argument, described above. **Warning**: this is experimental (see
            warning above for more details).

        edges: DataFrame or dict of hashable to DataFrame
            An edge list for each type of edges as a Pandas DataFrame containing a source, target
            and (optionally) weight column (the names of each are taken from the ``source_column``,
            ``target_column`` and ``edge_weight_label`` parameters). If there is only one type of
            edges, a DataFrame can be passed directly, and the type defaults to the
            ``edge_type_default`` parameter. Edges have an ID taken from the index of the dataframe,
            and they have to be unique across all types. This must be used with the ``nodes``
            argument. This uses the same basic structure as the ``node_features`` argument,
            described above. **Warning**: this is experimental (see warning above for more details).

        source_column: str, optional
            The name of the column to use as the source node of edges in the ``edges`` edge list
            argument.

        target_column: str, optional
            The name of the column to use as the target node of edges in the ``edges`` edge list
            argument.

        edge_weight_label: str, optional
            The name of the attribute to use as the weight of edges (for the `nodes`/`edges`
            DataFrame parameters, this is the name of the column to use).
    """

    def __init__(
        self,
        graph=None,
        is_directed=False,
        edge_weight_label="weight",
        node_type_name=globalvar.TYPE_ATTR_NAME,
        edge_type_name=globalvar.TYPE_ATTR_NAME,
        node_type_default=globalvar.NODE_TYPE_DEFAULT,
        edge_type_default=globalvar.EDGE_TYPE_DEFAULT,
        feature_name=globalvar.FEATURE_ATTR_NAME,
        target_name=globalvar.TARGET_ATTR_NAME,
        node_features=None,
        dtype="float32",
        nodes=None,
        edges=None,
        source_column=globalvar.SOURCE,
        target_column=globalvar.TARGET,
    ):
        if graph is not None:
            nodes, edges = convert.from_networkx(
                graph,
                node_type_name=node_type_name,
                edge_type_name=edge_type_name,
                node_type_default=node_type_default,
                edge_type_default=edge_type_default,
                edge_weight_label="weight",
                node_features=node_features,
                dtype=dtype,
            )
        else:
            warnings.warn(
                "StellarGraph(nodes=..., edges=...) is experimental: it has not been fully "
                "validated. It may be difficult to use and may have major changes at any time.",
                ExperimentalWarning,
            )

        if nodes is None:
            nodes = {}
        if edges is None:
            edges = {}

        self._is_directed = is_directed
        self._nodes = convert.convert_nodes(
            nodes, name="nodes", default_type=node_type_default, dtype=dtype,
        )
        self._edges = convert.convert_edges(
            edges,
            self._nodes,
            name="edges",
            default_type=edge_type_default,
            source_column=source_column,
            target_column=target_column,
            weight_column=edge_weight_label,
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

    def nodes(self) -> Iterable[Any]:
        """
        Obtains the collection of nodes in the graph.

        Returns:
            The graph nodes.
        """
        return self._nodes.ids.pandas_index

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
            return self.neighbors(
                node, include_edge_weight=include_edge_weight, edge_types=edge_types
            )

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
            return self.neighbors(
                node, include_edge_weight=include_edge_weight, edge_types=edge_types
            )

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
            node_type (hashable, optional): a type of nodes that exist in the graph

        Returns:
            A list of node IDs with type node_type
        """
        if node_type is None:
            return self.nodes()

        ilocs = self._nodes.type_range(node_type)
        return list(self._nodes.ids.from_iloc(ilocs))

    def _key_error_for_missing(self, query_ids, node_ilocs):
        valid = self._nodes.ids.is_valid(node_ilocs)
        missing_indices = np.where(~valid)
        missing_values = np.asarray(query_ids)[missing_indices]
        return KeyError(missing_values)

    def node_type(self, node):
        """
        Get the type of the node

        Args:
            node: Node ID

        Returns:
            Node type
        """
        is_single = not is_real_iterable(node)
        if is_single:
            node = np.array([node])

        node_ilocs = self._nodes.ids.to_iloc(node)

        try:
            types = self._nodes.type_of_iloc(node_ilocs)
        except IndexError:
            raise self._key_error_for_missing(node, node_ilocs)

        if is_single:
            return types[0]

        # FIXME: we need to choose the type to return here properly
        raise NotImplementedError(
            "passing multiple node IDs to node_type is not supported"
        )

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
        all_sizes = self._nodes.feature_sizes()
        if node_types is None:
            return all_sizes

        node_types = set(node_types)
        return {
            type_name: size for type_name, size in all_sizes if type_name in node_types
        }

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

    def node_features(self, nodes, node_type=None):
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

        nones = nodes == None
        if not (nones | valid).all():
            # every ID should be either valid or None, otherwise it was a completely unknown ID
            raise self._key_error_for_missing(nodes[~nones], node_ilocs[~nones])

        sampled = self._nodes.features(node_type, valid_ilocs)
        features = np.zeros((len(nodes), sampled.shape[1]))
        features[valid] = sampled

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

    def info(self, show_attributes=True, sample=None):
        """
        Return an information string summarizing information on the current graph.
        This includes node and edge type information and their attributes.

        Note: This requires processing all nodes and edges and could take a long
        time for a large graph.

        Args:
            show_attributes (bool, default True): If True, include attributes information
            sample (int): To speed up the graph analysis, use only a random sample of
                          this many nodes and edges.

        Returns:
            An information string.
        """
        directed_str = "Directed" if self.is_directed() else "Undirected"
        lines = [
            f"{type(self)}: {directed_str} multigraph",
            f" Nodes: {self.number_of_nodes()}, Edges: {self.number_of_edges()}",
        ]

        # Numpy processing is much faster than NetworkX processing, so we don't bother sampling.
        gs = self.create_graph_schema()

        def str_edge_type(et):
            n1, rel, n2 = et
            return f"{n1}-{rel}->{n2}"

        lines.append("")
        lines.append(" Node types:")

        for nt in gs.node_types:
            nodes = self.nodes_of_type(nt)
            lines.append(f"  {nt}: [{len(nodes)}]")
            edge_types = ", ".join(str_edge_type(et) for et in gs.schema[nt])
            lines.append(f"    Edge types: {edge_types}")

        lines.append("")
        lines.append(" Edge types:")

        # FIXME: it would be better for the schema to just include the counts directly
        for src_ty, rel_ty, tgt_ty, count in self._unique_type_triples(
            return_counts=True
        ):
            et = EdgeType(src_ty, rel_ty, tgt_ty)
            lines.append(f"    {str_edge_type(et)}: [{count}]")

        return "\n".join(lines)

    def create_graph_schema(self, create_type_maps=None, nodes=None):
        """
        Create graph schema in dict of dict format from current graph.

        Note the assumption we make that there is only one
        edge of a particular edge type per node pair.

        This means that specifying an edge by node0, node1 and edge type
        is unique.

        Arguments:
            nodes (list): A list of node IDs to use to build schema. This must
                represent all node types and all edge types in the graph.
                If not specified, all nodes and edges in the graph are used.

        Returns:
            GraphSchema object.
        """
        if create_type_maps is not None:
            warnings.warn(
                "The 'create_type_maps' parameter is ignored now, and does not need to be specified",
                DeprecationWarning,
            )

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

    def to_adjacency_matrix(self, nodes: Optional[Iterable] = None):
        """
        Obtains a SciPy sparse adjacency matrix of edge weights.

        Args:
            nodes (iterable): The optional collection of nodes
                comprising the subgraph. If specified, then the
                adjacency matrix is computed for the subgraph;
                otherwise, it is computed for the full graph.

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

        weights = self._edges.weights[selector]
        # these indices are computed relative to the index above. If `nodes` is None, they'll be the
        # overall ilocs (for the original graph), otherwise they'll be the indices of the `nodes`
        # list.
        src_idx = index.to_iloc(self._edges.sources[selector])
        tgt_idx = index.to_iloc(self._edges.targets[selector])
        n = len(index)

        adj = sps.csr_matrix(
            (weights, (src_idx, tgt_idx)), shape=(n, n), dtype="float32"
        )
        if not self.is_directed():
            # in an undirected graph, the adjacency matrix should be symmetric: which means counting
            # weights from either "incoming" or "outgoing" edges, but not double-counting self loops
            backward = sps.csr_matrix(
                (weights, (tgt_idx, src_idx)), shape=(n, n), dtype="float32"
            )
            backward.setdiag(0)
            adj += backward

        # this is a multigraph, let's eliminate any duplicate entries
        adj.sum_duplicates()
        return adj

    def to_networkx(
        self,
        node_type_name=globalvar.TYPE_ATTR_NAME,
        edge_type_name=globalvar.TYPE_ATTR_NAME,
        edge_weight_label=globalvar.WEIGHT,
        feature_name=globalvar.FEATURE_ATTR_NAME,
    ):
        """
        Create a NetworkX MultiGraph or MultiDiGraph instance representing this graph.

        Returns:
             An instance of `networkx.MultiDiGraph` (if directed) or `networkx.MultiGraph` (if
             undirected) containing all the nodes & edges and their types & features in this graph.
        """
        import networkx

        if self.is_directed():
            graph = networkx.MultiDiGraph()
        else:
            graph = networkx.MultiGraph()

        for ty in self.node_types:
            node_ids = self.nodes_of_type(ty)
            ty_dict = {node_type_name: ty}

            features = self.node_features(node_ids, node_type=ty)

            for node_id, node_features in zip(node_ids, features):
                graph.add_node(
                    node_id, **ty_dict, **{feature_name: node_features},
                )

        iterator = zip(
            self._edges.sources,
            self._edges.targets,
            self._edges.type_of_iloc(slice(None)),
            self._edges.weights,
        )
        graph.add_edges_from(
            (src, dst, {edge_type_name: type_, edge_weight_label: weight})
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
        return self._nodes._id_index.to_iloc(nodes)

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
        graph=None,
        edge_weight_label="weight",
        node_type_name=globalvar.TYPE_ATTR_NAME,
        edge_type_name=globalvar.TYPE_ATTR_NAME,
        node_type_default=globalvar.NODE_TYPE_DEFAULT,
        edge_type_default=globalvar.EDGE_TYPE_DEFAULT,
        feature_name=globalvar.FEATURE_ATTR_NAME,
        target_name=globalvar.TARGET_ATTR_NAME,
        node_features=None,
        dtype="float32",
        nodes=None,
        edges=None,
        source_column=globalvar.SOURCE,
        target_column=globalvar.TARGET,
    ):
        super().__init__(
            graph=graph,
            is_directed=True,
            edge_weight_label=edge_weight_label,
            node_type_name=node_type_name,
            edge_type_name=edge_type_name,
            node_type_default=node_type_default,
            edge_type_default=edge_type_default,
            feature_name=feature_name,
            target_name=target_name,
            node_features=node_features,
            dtype=dtype,
            nodes=nodes,
            edges=edges,
            source_column=source_column,
            target_column=target_column,
        )
