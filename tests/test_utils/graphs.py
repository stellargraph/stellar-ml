# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
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


from stellargraph import StellarGraph, StellarDiGraph, IndexedArray
import networkx as nx
import pandas as pd
import numpy as np
import random
import pytest


def create_graph_features():
    # APPNP, ClusterGCN, GCN, PPNP, node_mappers, full_batch_generators
    features = np.array([[1, 1], [1, 0], [0, 1]])
    nodes = pd.DataFrame(features, index=["a", "b", "c"])
    edges = pd.DataFrame(
        [("a", "b"), ("b", "c"), ("a", "c")], columns=["source", "target"]
    )
    return StellarGraph(nodes, edges), features


def relational_create_graph_features(is_directed=False, edge_weights=False):
    # RGCN, relational node mappers
    r1 = {"label": "r1"}
    r2 = {"label": "r2"}
    features = np.array([[1, 1], [1, 0], [0, 1]])
    nodes = pd.DataFrame(features, index=["a", "b", "c"])
    edges = {
        "r1": pd.DataFrame([("a", "b"), ("b", "c")], columns=["source", "target"]),
        "r2": pd.DataFrame([("a", "c")], columns=["source", "target"], index=[2]),
    }
    if edge_weights:
        edges["r1"]["weight"] = [2.0, 0.5]

    SG = StellarDiGraph if is_directed else StellarGraph
    return SG(nodes, edges), features


def example_graph_nx(
    feature_size=None, label="default", feature_name="feature", is_directed=False
):
    graph = nx.DiGraph() if is_directed else nx.Graph()
    elist = [(1, 2), (2, 3), (1, 4), (4, 2)]
    graph.add_nodes_from([1, 2, 3, 4], label=label)
    graph.add_edges_from(elist, label=label)

    # Add example features
    if feature_size is not None:
        for v in graph.nodes():
            graph.nodes[v][feature_name] = int(v) * np.ones(feature_size)

    return graph


def repeated_features(values_to_repeat, width):
    if width is None:
        return None

    if isinstance(width, int):
        width = (width,)

    values = np.asarray(values_to_repeat, dtype=np.float32)
    column = values.reshape(values.shape + (1,) * len(width))
    return np.tile(column, width)


def example_graph(
    feature_size=None,
    node_label="default",
    edge_label="default",
    feature_name="feature",
    is_directed=False,
    edge_feature_size=None,
    edge_weights=False,
):
    elist = pd.DataFrame([(1, 2), (2, 3), (1, 4), (4, 2)], columns=["source", "target"])
    if edge_feature_size is not None:
        edge_features = repeated_features(-elist.index, edge_feature_size)
        elist = elist.join(pd.DataFrame(edge_features))
    if edge_weights:
        elist["weight"] = [0.1, 1.0, 20.0, 1.3]

    nodes = [1, 2, 3, 4]
    node_features = repeated_features(nodes, feature_size)

    nodes = IndexedArray(node_features, index=nodes)

    cls = StellarDiGraph if is_directed else StellarGraph
    return cls(nodes={node_label: nodes}, edges={edge_label: elist})


def example_hin_1_nx(feature_name=None, for_nodes=None, feature_sizes=None):
    # stellargraph
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3], label="A")
    graph.add_nodes_from([4, 5, 6], label="B")
    graph.add_edges_from([(0, 4), (1, 4), (1, 5), (2, 4), (3, 5)], label="R")
    graph.add_edges_from([(4, 5)], label="F")

    if feature_name is not None:
        if for_nodes is None:
            for_nodes = list(graph.nodes())

        if feature_sizes is None:
            feature_sizes = dict()

        for v in for_nodes:
            fs = feature_sizes.get(graph.nodes[v]["label"], 10)
            graph.nodes[v][feature_name] = v * np.ones(fs)

    return graph


def example_hin_1(
    feature_sizes=None,
    is_directed=False,
    self_loop=False,
    reverse_order=False,
    edge_features=False,
) -> StellarGraph:
    def features(label, ids):
        if feature_sizes is None:
            return None
        else:
            feature_size = feature_sizes.get(label, 10)
            return repeated_features(ids, feature_size)

    a_ids = [0, 1, 2, 3]
    if reverse_order:
        a_ids = a_ids[::-1]
    a = IndexedArray(features("A", a_ids), index=a_ids)

    b_ids = [4, 5, 6]
    if reverse_order:
        b_ids = b_ids[::-1]
    b = IndexedArray(features("B", b_ids), index=b_ids)

    r_edges = [(4, 0), (1, 5), (1, 4), (2, 4), (5, 3)]
    f_edges, f_index = [(4, 5)], [100]
    if self_loop:
        # make it a multigraph, across types and within a single one
        r_edges.append((5, 5))
        f_edges.extend([(5, 5), (5, 5)])
        f_index.extend([101, 102])

    r = pd.DataFrame(r_edges, columns=["source", "target"])

    # add some weights for the f edges, but not others
    f_columns = ["source", "target", "weight"]
    for i, src_tgt in enumerate(f_edges):
        f_edges[i] = src_tgt + (10 + i,)

    f = pd.DataFrame(f_edges, columns=f_columns, index=f_index)

    if edge_features:
        r = r.join(pd.DataFrame(-features("R", r.index), index=r.index))
        f = f.join(pd.DataFrame(-features("F", f.index), index=f.index))

    cls = StellarDiGraph if is_directed else StellarGraph
    return cls(nodes={"A": a, "B": b}, edges={"R": r, "F": f})


def create_test_graph(is_directed=False):
    # biased random walker, breadth first walker, directed breadth first walker, uniform random walker

    nodes = pd.DataFrame(
        index=["0", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "self loner", "loner"]
    )
    edges = pd.DataFrame(
        [
            ("0", 1),
            ("0", 2),
            (1, 3),
            (1, 4),
            (3, 6),
            (4, 7),
            (4, 8),
            (2, 5),
            (5, 9),
            (5, 10),
            ("0", "0"),
            (1, 1),
            (3, 3),
            (6, 6),
            (4, 4),
            (7, 7),
            (8, 8),
            (2, 2),
            (5, 5),
            (9, 9),
            ("self loner", "self loner"),  # an isolated node with a self link
        ],
        columns=["source", "target"],
    )
    cls = StellarDiGraph if is_directed else StellarGraph
    return cls(nodes, edges)


def example_graph_1_saliency_maps(feature_size=None):
    # saliency gcn, saliency gat
    nlist = [0, 1, 2, 3, 4]
    if feature_size is None:
        nodes = pd.DataFrame(index=nlist)
    else:
        # Example features
        nodes = pd.DataFrame(np.ones((len(nlist), feature_size)), index=nlist)

    elist = [(0, 1), (0, 2), (2, 3), (3, 4), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    edges = pd.DataFrame(elist, columns=["source", "target"])

    return StellarGraph(nodes, edges)


def example_graph_random(
    feature_size=4,
    n_edges=20,
    n_nodes=6,
    n_isolates=1,
    is_directed=False,
    node_types=1,
    edge_types=1,
):
    # core/utils, link mapper, node mapper graph 3

    if is_directed:
        cls = StellarDiGraph
    else:
        cls = StellarGraph

    default_feature_size = 0
    if not isinstance(feature_size, dict):
        if isinstance(feature_size, int):
            default_feature_size = feature_size
        feature_size = {}

    def nodes(node_type):
        node_ids = range(n_nodes * node_type, n_nodes * (node_type + 1))
        feat_dim = feature_size.get(node_type, default_feature_size)
        features = np.random.random((len(node_ids), feat_dim))
        return pd.DataFrame(features, index=node_ids)

    n_noniso = n_nodes - n_isolates

    def edges(edge_type):
        edge_ids = range(n_edges * edge_type, n_edges * (edge_type + 1))
        # separately compute the within-type and type part of the node IDs for each edge, so that
        # each type has some isolated nodes (rather than just the last type)
        within_type = np.random.randint(0, n_noniso, size=(n_edges, 2))
        which_type = np.random.randint(0, node_types, size=within_type.shape)
        return pd.DataFrame(
            within_type + which_type * n_nodes,
            columns=["source", "target"],
            index=edge_ids,
        )

    # string node types are more realistic
    return cls(
        {f"n-{nt}": nodes(nt) for nt in range(node_types)},
        {f"e-{et}": edges(et) for et in range(edge_types)},
    )


def node_features(seed=0) -> pd.DataFrame:
    random = np.random.RandomState(seed)
    node_data_np = random.rand(10, 10)
    return pd.DataFrame(node_data_np)


@pytest.fixture
def petersen_graph() -> StellarGraph:
    nxg = nx.petersen_graph()
    return StellarGraph.from_networkx(nxg, node_features=node_features())


@pytest.fixture
def line_graph() -> StellarGraph:
    nodes = node_features()
    edges = pd.DataFrame([(i, i + 1) for i in range(9)], columns=["source", "target"])
    return StellarGraph(nodes, edges)


@pytest.fixture
def knowledge_graph():
    nodes = ["a", "b", "c", "d"]

    edge_counter = 0

    def edge_df(*elements):
        nonlocal edge_counter
        end = edge_counter + len(elements)
        index = range(edge_counter, end)
        edge_counter = end
        return pd.DataFrame(elements, columns=["source", "target"], index=index)

    edges = {
        "W": edge_df(("a", "b")),
        "X": edge_df(("a", "b"), ("b", "c")),
        "Y": edge_df(("b", "a")),
        "Z": edge_df(("d", "b")),
    }

    return StellarDiGraph(nodes=pd.DataFrame(index=nodes), edges=edges)


@pytest.fixture
def tree_graph() -> StellarGraph:
    nodes = pd.DataFrame(index=["root", "0", 1, 2, "c1.1", "c2.1", "c2.2"])
    edges = pd.DataFrame(
        [
            ("root", 2),
            ("root", 1),
            ("root", "0"),
            (2, "c2.1"),
            (2, "c2.2"),
            (1, "c1.1"),
        ],
        columns=["source", "target"],
    )

    return StellarDiGraph(nodes, edges)


import itertools as it
values = list(it.product(range(2, 10 + 1), range(1, 11 + 1)))
#values = [(2, 3)]
@pytest.fixture(params=values, ids=[f"{m1},{m2},{2*m1+m2}" for m1, m2 in values])
def barbell(request):
    m1, m2 = request.param
    graph = nx.barbell_graph(m1=m1, m2=m2)
    for i, (src, tgt) in enumerate(graph.edges):
        graph[src][tgt]["weight"] = i + 2**(i % 6)
    return StellarGraph.from_networkx(graph)


@pytest.fixture
def weighted_hin():
    a_ids = [0, 1, 2, 3]
    a = pd.DataFrame(index=a_ids)

    b_ids = [4, 5, 6]
    b = pd.DataFrame(index=b_ids)

    # no weights A-R->A
    r_ids = [7, 8]
    r = pd.DataFrame([(0, 1), (0, 2)], columns=["source", "target"], index=r_ids)

    # single weighted edge A-S->A
    s_ids = [9, 10]
    s = pd.DataFrame([(0, 3, 2)], columns=["source", "target", "weight"], index=s_ids)

    # 3 edges with same weight A-T->B
    t_ids = [11, 12, 13]
    t = pd.DataFrame(
        [(0, 4, 2), (0, 5, 2), (0, 6, 2)],
        columns=["source", "target", "weight"],
        index=t_ids,
    )

    # weights [2, 3] A-U->A; weights [4, 5, 6] A-U->B
    u_ids = [14, 15, 16, 17, 18]
    u = pd.DataFrame(
        [(1, 2, 2), (1, 3, 3), (1, 4, 4), (1, 4, 5), (6, 1, 5)],
        columns=["source", "target", "weight"],
        index=u_ids,
    )

    return StellarGraph(nodes={"A": a, "B": b}, edges={"R": r, "S": s, "T": t, "U": u})


def weighted_tree(is_directed=False):
    # a binary tree:
    # 0---1--3
    # |   |
    # |   4
    # 2--5
    # |
    # 6
    nodes = repeated_features(range(7), 3)
    edges = pd.DataFrame(
        {
            "source": [0, 0, 1, 1, 2, 2],
            "target": [1, 2, 3, 4, 5, 6],
            "weight": [1.0, 5, 20, 1, 1, 0],
        }
    )
    cls = StellarDiGraph if is_directed else StellarGraph

    def check_occurrence(sequence):
        from collections import Counter

        occurrence = Counter(sequence)
        # 0--2 has higher weight than 0--1
        assert occurrence[2] > occurrence[1]
        # 1--3 has higher weight than 1--4
        assert occurrence[3] > occurrence[4]
        # 2--6 has 0 weight (i.e. should never be taken)
        assert 6 not in occurrence

    return cls(nodes, edges), check_occurrence
