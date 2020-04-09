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
Mapper tests:

"""
from stellargraph.core.graph import *
from stellargraph.mapper import *

import networkx as nx
import numpy as np
import random
import pytest
import pandas as pd
import scipy.sparse as sps
from ..test_utils.graphs import (
    example_graph,
    example_graph_random,
    example_hin_1,
    create_graph_features,
    repeated_features,
)
from .. import test_utils


pytestmark = test_utils.ignore_stellargraph_experimental_mark


# FIXME (#535): Consider using graph fixtures
def example_graph_2(feature_size=None):
    nlist = [1, 2, 3, 4, 5]
    nodes = pd.DataFrame(repeated_features(nlist, feature_size), index=nlist)

    elist = [(1, 2), (1, 3), (1, 4), (3, 2), (3, 5)]
    edges = pd.DataFrame(elist, columns=["source", "target"])

    return StellarGraph(nodes, edges)


def example_hin_2(feature_size_by_type=None):
    if feature_size_by_type is None:
        feature_size_by_type = {"t1": None, "t2": None}

    nodes_type_1 = [0, 1, 2, 3]
    nodes_type_2 = [4, 5]
    nodes = {
        "t1": pd.DataFrame(
            repeated_features(nodes_type_1, feature_size_by_type["t1"]),
            index=nodes_type_1,
        ),
        "t2": pd.DataFrame(
            repeated_features(nodes_type_2, feature_size_by_type["t2"]),
            index=nodes_type_2,
        ),
    }
    edges = {
        "e1": pd.DataFrame(
            [(0, 4), (1, 4), (2, 5), (3, 5)], columns=["source", "target"]
        )
    }

    return StellarGraph(nodes, edges), nodes_type_1, nodes_type_2


def example_hin_3(feature_size_by_type=None):
    if feature_size_by_type is None:
        feature_size_by_type = {"t1": None, "t2": None}

    nodes_type_1 = np.array([0, 1, 2])
    nodes_type_2 = np.array([4, 5, 6])
    nodes = {
        "t1": pd.DataFrame(
            repeated_features(10 + nodes_type_1, feature_size_by_type["t1"]),
            index=nodes_type_1,
        ),
        "t2": pd.DataFrame(
            repeated_features(10 + nodes_type_2, feature_size_by_type["t2"]),
            index=nodes_type_2,
        ),
    }
    edges = {
        "e1": pd.DataFrame([(0, 4), (1, 5)], columns=["source", "target"]),
        "e2": pd.DataFrame([(0, 2)], columns=["source", "target"], index=[2]),
    }

    # Node 2 has no edges of type 1
    # Node 1 has no edges of type 2
    # Node 6 has no edges

    return StellarGraph(nodes, edges), nodes_type_1, nodes_type_2


def test_nodemapper_constructor_nx():
    """
    GraphSAGENodeGenerator requires a StellarGraph object
    """
    G = nx.Graph()
    G.add_nodes_from(range(4))

    with pytest.raises(TypeError):
        GraphSAGENodeGenerator(G, batch_size=2, num_samples=[2, 2])


def test_nodemapper_constructor_no_feats():
    """
    GraphSAGENodeGenerator requires the graph to have features
    """
    n_feat = 4

    G = example_graph()
    with pytest.raises(RuntimeError):
        GraphSAGENodeGenerator(G, batch_size=2, num_samples=[2, 2])


def test_nodemapper_constructor():
    n_feat = 4

    G = example_graph(feature_size=n_feat)

    generator = GraphSAGENodeGenerator(G, batch_size=2, num_samples=[2, 2])

    mapper = generator.flow(list(G.nodes()))

    assert generator.batch_size == 2
    assert mapper.data_size == 4
    assert len(mapper.ids) == 4


def test_nodemapper_1():
    n_feat = 4
    n_batch = 2

    # test graph
    G1 = example_graph(n_feat)

    mapper1 = GraphSAGENodeGenerator(G1, batch_size=n_batch, num_samples=[2, 2]).flow(
        G1.nodes()
    )
    assert len(mapper1) == 2

    G2 = example_graph_2(n_feat)

    mapper2 = GraphSAGENodeGenerator(G2, batch_size=n_batch, num_samples=[2, 2]).flow(
        G2.nodes()
    )
    assert len(mapper2) == 3

    for mapper in [mapper1, mapper2]:
        for ii in range(2):
            nf, nl = mapper[ii]
            assert len(nf) == 3
            assert nf[0].shape == (n_batch, 1, n_feat)
            assert nf[1].shape == (n_batch, 2, n_feat)
            assert nf[2].shape == (n_batch, 2 * 2, n_feat)
            assert nl is None

    # Check beyond the graph lengh
    with pytest.raises(IndexError):
        nf, nl = mapper1[len(mapper1)]

    # Check the last batch
    nf, nl = mapper2[len(mapper2) - 1]
    assert nf[0].shape == (1, 1, n_feat)
    assert nf[1].shape == (1, 2, n_feat)
    assert nf[2].shape == (1, 2 * 2, n_feat)

    # This will fail as the nodes are not in the graph
    with pytest.raises(KeyError):
        GraphSAGENodeGenerator(G1, batch_size=2, num_samples=[2, 2]).flow(["A", "B"])


@pytest.mark.parametrize("shuffle", [True, False])
def test_nodemapper_shuffle(shuffle):
    n_feat = 1
    n_batch = 2

    G = example_graph_2(feature_size=n_feat)
    nodes = list(G.nodes())

    def flatten_features(seq):
        # check (features == labels) and return flattened features
        batches = [
            (np.ravel(seq[i][0][0]), np.array(seq[i][1])) for i in range(len(seq))
        ]
        features, labels = zip(*batches)
        features, labels = np.concatenate(features), np.concatenate(labels)
        assert all(features == labels)
        return features

    def consecutive_epochs(seq):
        features = flatten_features(seq)
        seq.on_epoch_end()
        features_next = flatten_features(seq)
        return features, features_next

    seq = GraphSAGENodeGenerator(G, batch_size=n_batch, num_samples=[0]).flow(
        nodes, nodes, shuffle=shuffle
    )

    max_iter = 5
    comparison_results = set()

    for i in range(max_iter):
        f1, f2 = consecutive_epochs(seq)
        comparison_results.add(all(f1 == f2))

    if not shuffle:
        assert comparison_results == {True}
    else:
        assert False in comparison_results


def test_nodemapper_with_labels():
    n_feat = 4
    n_batch = 2

    # test graph
    G2 = example_graph_2(n_feat)
    nodes = list(G2.nodes())
    labels = [n * 2 for n in nodes]

    gen = GraphSAGENodeGenerator(G2, batch_size=n_batch, num_samples=[2, 2]).flow(
        nodes, labels
    )
    assert len(gen) == 3

    for ii in range(3):
        nf, nl = gen[ii]

        # Check sizes - note batch sizes are (2,2,1) for each iteration
        assert len(nf) == 3
        assert nf[0].shape[1:] == (1, n_feat)
        assert nf[1].shape[1:] == (2, n_feat)
        assert nf[2].shape[1:] == (2 * 2, n_feat)

        # Check labels
        assert all(int(a) == int(2 * b) for a, b in zip(nl, nf[0][:, 0, 0]))

    # Check beyond the graph lengh
    with pytest.raises(IndexError):
        nf, nl = gen[len(gen)]


def test_nodemapper_zero_samples():
    n_feat = 4
    n_batch = 2

    # test graph
    G = example_graph(feature_size=n_feat)
    mapper = GraphSAGENodeGenerator(G, batch_size=n_batch, num_samples=[0]).flow(
        G.nodes()
    )

    # This is an edge case, are we sure we want this behaviour?
    assert len(mapper) == 2
    for ii in range(len(mapper)):
        nf, nl = mapper[ii]
        assert len(nf) == 2
        assert nf[0].shape == (n_batch, 1, n_feat)
        assert nf[1].shape == (n_batch, 0, n_feat)
        assert nl is None

    # test graph
    G = example_graph(feature_size=n_feat)
    mapper = GraphSAGENodeGenerator(G, batch_size=n_batch, num_samples=[0, 0]).flow(
        G.nodes()
    )

    # This is an edge case, are we sure we want this behaviour?
    assert len(mapper) == 2
    for ii in range(len(mapper)):
        nf, nl = mapper[ii]
        assert len(nf) == 3
        assert nf[0].shape == (n_batch, 1, n_feat)
        assert nf[1].shape == (n_batch, 0, n_feat)
        assert nf[1].shape == (n_batch, 0, n_feat)
        assert nl is None


def test_nodemapper_isolated_nodes():
    n_feat = 4
    n_batch = 2

    # test graph
    G = example_graph_random(feature_size=n_feat, n_nodes=6, n_isolates=1, n_edges=20)

    # Check connectedness
    Gnx = G.to_networkx()
    ccs = list(nx.connected_components(Gnx))
    assert len(ccs) == 2

    n_isolates = [5]
    assert nx.degree(Gnx, n_isolates[0]) == 0

    # Check both isolated and non-isolated nodes have same sampled feature shape
    for head_nodes in [[1], [2], n_isolates]:
        mapper = GraphSAGENodeGenerator(G, batch_size=n_batch, num_samples=[2, 2]).flow(
            head_nodes
        )
        nf, nl = mapper[0]
        assert nf[0].shape == (1, 1, n_feat)
        assert nf[1].shape == (1, 2, n_feat)
        assert nf[2].shape == (1, 4, n_feat)

    # One isolate and one non-isolate
    mapper = GraphSAGENodeGenerator(G, batch_size=n_batch, num_samples=[2, 2]).flow(
        [1, 5]
    )
    nf, nl = mapper[0]
    assert nf[0].shape == (2, 1, n_feat)
    assert nf[1].shape == (2, 2, n_feat)
    assert nf[2].shape == (2, 4, n_feat)

    # Isolated nodes have the "dummy node" as neighbours
    # Currently, the dummy node has zeros for features – this could change
    assert pytest.approx(nf[1].numpy()[1]) == 0
    assert pytest.approx(nf[2].numpy()[2:]) == 0


def test_nodemapper_incorrect_targets():
    """
    Tests checks on target shape
    """
    n_feat = 4
    n_batch = 2

    # test graph
    G = example_graph(feature_size=n_feat)

    with pytest.raises(TypeError):
        GraphSAGENodeGenerator(G, batch_size=n_batch, num_samples=[0]).flow(
            list(G.nodes()), 1
        )

    with pytest.raises(ValueError):
        GraphSAGENodeGenerator(G, batch_size=n_batch, num_samples=[0]).flow(
            list(G.nodes()), targets=[]
        )


def test_hinnodemapper_constructor():
    feature_sizes = {"A": 10, "B": 10}
    G = example_hin_1(feature_sizes=feature_sizes)

    # Should fail when head nodes are of different type
    with pytest.raises(ValueError):
        HinSAGENodeGenerator(
            G, batch_size=2, num_samples=[2, 2], head_node_type="A"
        ).flow(G.nodes())

    gen = HinSAGENodeGenerator(G, batch_size=2, num_samples=[2, 2], head_node_type="A")
    mapper = gen.flow([0, 1, 2, 3])
    assert gen.batch_size == 2
    assert mapper.data_size == 4
    assert len(mapper.ids) == 4


def test_hinnodemapper_constructor_all_options():
    feature_sizes = {"A": 10, "B": 10}
    G = example_hin_1(feature_sizes=feature_sizes)

    gen = HinSAGENodeGenerator(G, batch_size=2, num_samples=[2, 2], head_node_type="A")

    nodes_of_type_a = G.nodes(node_type="A")
    mapper = gen.flow(nodes_of_type_a)
    assert gen.batch_size == 2
    assert mapper.data_size == len(nodes_of_type_a)


def test_hinnodemapper_constructor_no_features():
    G = example_hin_1(feature_sizes=None)
    with pytest.raises(RuntimeError):
        mapper = HinSAGENodeGenerator(
            G, batch_size=2, num_samples=[2, 2], head_node_type="A"
        ).flow(G.nodes())


def test_hinnodemapper_constructor_nx_graph():
    G = nx.Graph()
    with pytest.raises(TypeError):
        HinSAGENodeGenerator(G, batch_size=2, num_samples=[2, 2])

    with pytest.raises(TypeError):
        HinSAGENodeGenerator(None, batch_size=2, num_samples=[2, 2])


def test_hinnodemapper_level_1():
    batch_size = 2
    feature_sizes = {"t1": 1, "t2": 2}
    G, nodes_type_1, nodes_type_2 = example_hin_2(feature_sizes)

    mapper = HinSAGENodeGenerator(
        G, batch_size=batch_size, num_samples=[2], head_node_type="t2"
    ).flow(nodes_type_2)

    schema = G.create_graph_schema()
    sampling_adj = schema.type_adjacency_list(["t2"], 1)

    assert len(mapper) == 1

    # Get a batch!
    batch_feats, batch_targets = mapper[0]

    # Check shapes are (batch_size, nsamples, feature_size)
    assert np.shape(batch_feats[0]) == (2, 1, 2)
    assert np.shape(batch_feats[1]) == (2, 2, 1)

    # Check the types
    assert np.all(batch_feats[0] >= 4)
    assert np.all(batch_feats[1] < 4)


def test_hinnodemapper_level_2():
    batch_size = 2
    feature_sizes = {"t1": 1, "t2": 2}
    G, nodes_type_1, nodes_type_2 = example_hin_2(feature_sizes)

    mapper = HinSAGENodeGenerator(
        G, batch_size=batch_size, num_samples=[2, 3], head_node_type="t2"
    ).flow(nodes_type_2)

    schema = G.create_graph_schema()
    sampling_adj = schema.type_adjacency_list(["t2"], 2)

    assert len(mapper) == 1

    # Get a batch!
    batch_feats, batch_targets = mapper[0]

    # Check types match adjacency list
    assert len(batch_feats) == len(sampling_adj)
    for bf, adj in zip(batch_feats, sampling_adj):
        nt = adj[0]
        assert bf.shape[0] == batch_size
        assert bf.shape[2] == feature_sizes[nt]

        batch_node_types = {G.node_type(n) for n in np.ravel(bf)}

        assert len(batch_node_types) == 1
        assert nt in batch_node_types


def test_hinnodemapper_shuffle():
    random.seed(10)

    batch_size = 2
    feature_sizes = {"t1": 1, "t2": 4}
    G, nodes_type_1, nodes_type_2 = example_hin_2(feature_sizes)

    mapper = HinSAGENodeGenerator(
        G, batch_size=batch_size, num_samples=[0], head_node_type="t1"
    ).flow(nodes_type_1, nodes_type_1, shuffle=True)

    expected_node_batches = [[3, 2], [1, 0]]
    assert len(mapper) == 2
    for ii in range(len(mapper)):
        nf, nl = mapper[ii]
        assert all(np.ravel(nf[0]) == expected_node_batches[ii])
        assert all(np.array(nl) == expected_node_batches[ii])

    # This should re-shuffle the IDs
    mapper.on_epoch_end()
    expected_node_batches = [[2, 1], [3, 0]]
    assert len(mapper) == 2
    for ii in range(len(mapper)):
        nf, nl = mapper[ii]
        assert all(np.ravel(nf[0]) == expected_node_batches[ii])
        assert all(np.array(nl) == expected_node_batches[ii])

    # With no shuffle
    mapper = HinSAGENodeGenerator(
        G, batch_size=batch_size, num_samples=[0], head_node_type="t1"
    ).flow(nodes_type_1, nodes_type_1, shuffle=False)
    expected_node_batches = [[0, 1], [2, 3]]
    assert len(mapper) == 2
    for ii in range(len(mapper)):
        nf, nl = mapper[ii]
        assert all(np.ravel(nf[0]) == expected_node_batches[ii])
        assert all(np.array(nl) == expected_node_batches[ii])


def test_hinnodemapper_with_labels():
    batch_size = 2
    feature_sizes = {"t1": 1, "t2": 2}
    G, nodes_type_1, nodes_type_2 = example_hin_2(feature_sizes)

    labels = [n * 2 for n in nodes_type_1]

    gen = HinSAGENodeGenerator(
        G, batch_size=batch_size, num_samples=[2, 3], head_node_type="t1"
    ).flow(nodes_type_1, labels, shuffle=False)
    assert len(gen) == 2

    for ii in range(2):
        nf, nl = gen[ii]

        # Check sizes of neighbours and features (in bipartite graph)
        assert len(nf) == 3
        assert nf[0].shape == (2, 1, 1)
        assert nf[1].shape == (2, 2, 2)
        assert nf[2].shape == (2, 2 * 3, 1)

        # Check labels
        assert all(int(a) == int(2 * b) for a, b in zip(nl, nf[0][:, 0, 0]))

    # Check beyond the graph lengh
    with pytest.raises(IndexError):
        nf, nl = gen[len(gen)]


def test_hinnodemapper_manual_schema():
    """
    Tests checks on head nodes
    """
    n_batch = 2
    feature_sizes = {"t1": 1, "t2": 2}
    G, nodes_type_1, nodes_type_2 = example_hin_2(feature_sizes)

    # Create manual schema
    schema = G.create_graph_schema()
    HinSAGENodeGenerator(
        G, schema=schema, batch_size=n_batch, num_samples=[1], head_node_type="t1"
    ).flow(nodes_type_1)


def test_hinnodemapper_zero_samples():
    batch_size = 3
    feature_sizes = {"t1": 1, "t2": 1}
    G, nodes_type_1, nodes_type_2 = example_hin_3(feature_sizes)

    mapper = HinSAGENodeGenerator(
        G, batch_size=batch_size, num_samples=[0, 0], head_node_type="t2"
    ).flow(nodes_type_2)

    schema = G.create_graph_schema()
    sampling_adj = schema.type_adjacency_list(["t2"], 2)

    assert len(mapper) == 1

    # Get a batch!
    batch_feats, batch_targets = mapper[0]
    assert len(batch_feats) == len(sampling_adj)


def test_hinnodemapper_no_neighbors():
    batch_size = 3
    feature_sizes = {"t1": 1, "t2": 1}
    G, nodes_type_1, nodes_type_2 = example_hin_3(feature_sizes)

    mapper = HinSAGENodeGenerator(
        G, batch_size=batch_size, num_samples=[2, 1], head_node_type="t2"
    ).flow(nodes_type_2)

    schema = G.create_graph_schema()
    sampling_adj = schema.type_adjacency_list(["t2"], 2)

    assert len(mapper) == 1

    # Get a batch!
    batch_feats, batch_targets = mapper[0]
    assert len(batch_feats) == len(sampling_adj)

    # Head nodes
    assert np.all(np.ravel(batch_feats[0]) == np.array([14, 15, 16]))

    # Next level - node 6 has no neighbours
    assert np.all(batch_feats[1][:, 0, 0] == np.array([10, 11, 0]))

    # Following level has two edge types
    # First edge type (e1): Node 0 has 4, node 1 has 5, and node 6 sampling has terminated
    assert np.all(batch_feats[2][:, 0, 0] == np.array([14, 15, 0]))

    # Second edge type (e2): Node 0 has 2, node 1 has none, and node 6 sampling has terminated
    assert np.all(batch_feats[3][:, 0, 0] == np.array([12, 0, 0]))


def test_attri2vec_nodemapper_constructor_nx():
    """
    Attri2VecNodeGenerator requires a StellarGraph object
    """
    G = nx.Graph()
    G.add_nodes_from(range(4))

    with pytest.raises(TypeError):
        Attri2VecNodeGenerator(G, batch_size=2)


def test_attri2vec_nodemapper_constructor_no_feats():
    """
    Attri2VecNodeGenerator requires the graph to have features
    """

    G = example_graph()
    with pytest.raises(RuntimeError):
        Attri2VecNodeGenerator(G, batch_size=2)


def test_attri2vec_nodemapper_constructor():
    n_feat = 4

    G = example_graph(feature_size=n_feat)

    generator = Attri2VecNodeGenerator(G, batch_size=2)

    mapper = generator.flow(list(G.nodes()))

    assert generator.batch_size == 2
    assert mapper.data_size == 4
    assert len(mapper.ids) == 4


def test_attri2vec_nodemapper_1():
    n_feat = 4
    n_batch = 2

    # test graph
    G1 = example_graph(n_feat)

    mapper1 = Attri2VecNodeGenerator(G1, batch_size=n_batch).flow(G1.nodes())
    assert len(mapper1) == 2

    G2 = example_graph_2(n_feat)

    mapper2 = Attri2VecNodeGenerator(G2, batch_size=n_batch).flow(G2.nodes())
    assert len(mapper2) == 3

    for mapper in [mapper1, mapper2]:
        for ii in range(2):
            nf, nl = mapper[ii]
            assert nf.shape == (n_batch, n_feat)
            assert nl is None

    # Check beyond the graph lengh
    with pytest.raises(IndexError):
        nf, nl = mapper1[len(mapper1)]

    # Check the last batch
    nf, nl = mapper2[len(mapper2) - 1]
    assert nf.shape == (1, n_feat)

    # This will fail as the nodes are not in the graph
    #    with pytest.raises(KeyError):
    Attri2VecNodeGenerator(G1, batch_size=2).flow(["A", "B"])


def test_attri2vec_nodemapper_2():
    n_feat = 1
    n_batch = 2

    G = example_graph_2(feature_size=n_feat)
    nodes = list(G.nodes())

    # With no shuffle
    mapper = Attri2VecNodeGenerator(G, batch_size=n_batch).flow(nodes)
    expected_node_batches = [[1, 2], [3, 4], [5]]
    assert len(mapper) == 3
    for ii in range(len(mapper)):
        nf, nl = mapper[ii]
        assert all(np.ravel(nf) == expected_node_batches[ii])


class Test_FullBatchNodeGenerator:
    """
    Tests of FullBatchNodeGenerator class
    """

    n_feat = 4
    target_dim = 5

    G = example_graph_random(feature_size=n_feat, n_nodes=6, n_isolates=1, n_edges=20)
    N = len(G.nodes())

    def test_generator_constructor(self):
        generator = FullBatchNodeGenerator(self.G)
        assert generator.Aadj.shape == (self.N, self.N)
        assert generator.features.shape == (self.N, self.n_feat)

    def test_generator_constructor_wrong_G_type(self):
        with pytest.raises(TypeError):
            generator = FullBatchNodeGenerator(nx.Graph())

    def test_generator_constructor_hin(self):
        feature_sizes = {"t1": 1, "t2": 1}
        Ghin, nodes_type_1, nodes_type_2 = example_hin_3(feature_sizes)
        with pytest.raises(TypeError):
            generator = FullBatchNodeGenerator(Ghin)

    def generator_flow(
        self,
        G,
        node_ids,
        node_targets,
        sparse=False,
        method="none",
        k=1,
        teleport_probability=0.1,
    ):
        generator = FullBatchNodeGenerator(
            G,
            sparse=sparse,
            method=method,
            k=k,
            teleport_probability=teleport_probability,
        )
        n_nodes = G.number_of_nodes()

        gen = generator.flow(node_ids, node_targets)
        if sparse:
            [X, tind, A_ind, A_val], y = gen[0]
            A_sparse = sps.coo_matrix(
                (A_val[0], (A_ind[0, :, 0], A_ind[0, :, 1])), shape=(n_nodes, n_nodes)
            )
            A_dense = A_sparse.toarray()

        else:
            [X, tind, A], y = gen[0]
            A_dense = A[0].numpy()

        assert np.allclose(X, gen.features)  # X should be equal to gen.features
        assert tind.shape[1] == len(node_ids)

        if node_targets is not None:
            assert np.allclose(y, node_targets)

        # Check that the diagonals are one
        if method == "self_loops":
            assert np.allclose(A_dense.diagonal(), 1)

        return A_dense, tind, y

    def test_generator_flow_notargets(self):
        node_ids = list(self.G.nodes())[:3]
        _, tind, y = self.generator_flow(
            self.G, node_ids, None, sparse=False, method="none"
        )
        assert np.allclose(tind, range(3))
        _, tind, y = self.generator_flow(
            self.G, node_ids, None, sparse=True, method="none"
        )
        assert np.allclose(tind, range(3))

        node_ids = list(self.G.nodes())
        _, tind, y = self.generator_flow(
            self.G, node_ids, None, sparse=False, method="none"
        )
        assert np.allclose(tind, range(len(node_ids)))
        _, tind, y = self.generator_flow(
            self.G, node_ids, None, sparse=True, method="none"
        )
        assert np.allclose(tind, range(len(node_ids)))

    def test_generator_flow_withtargets(self):
        node_ids = list(self.G.nodes())[:3]
        node_targets = np.ones((len(node_ids), self.target_dim)) * np.arange(3)[:, None]
        _, tind, y = self.generator_flow(self.G, node_ids, node_targets, sparse=True)
        assert np.allclose(tind, range(3))
        assert np.allclose(y, node_targets[:3])
        _, tind, y = self.generator_flow(self.G, node_ids, node_targets, sparse=False)
        assert np.allclose(tind, range(3))
        assert np.allclose(y, node_targets[:3])

        node_ids = list(self.G.nodes())[::-1]
        node_targets = (
            np.ones((len(node_ids), self.target_dim))
            * np.arange(len(node_ids))[:, None]
        )
        _, tind, y = self.generator_flow(self.G, node_ids, node_targets)
        assert np.allclose(tind, range(len(node_ids))[::-1])
        assert np.allclose(y, node_targets)

    def test_generator_flow_targets_as_list(self):
        generator = FullBatchNodeGenerator(self.G)
        node_ids = list(self.G.nodes())[:3]
        node_targets = [1] * len(node_ids)
        gen = generator.flow(node_ids, node_targets)

        inputs, y = gen[0]
        assert y.shape == (1, 3)
        assert np.sum(y) == 3

    def test_generator_flow_targets_not_iterator(self):
        generator = FullBatchNodeGenerator(self.G)
        node_ids = list(self.G.nodes())[:3]
        node_targets = 1
        with pytest.raises(TypeError):
            generator.flow(node_ids, node_targets)

    def test_fullbatch_generator_init_1(self):
        G, feats = create_graph_features()
        generator = FullBatchNodeGenerator(G, method=None)
        assert np.array_equal(feats, generator.features)

    def test_fullbatch_generator_init_3(self):
        G, _ = create_graph_features()
        func = "Not callable"

        with pytest.raises(ValueError):
            generator = FullBatchNodeGenerator(G, "test", transform=func)

    def test_fullbatch_generator_transform(self):
        G, _ = create_graph_features()

        def func(features, A, **kwargs):
            return features, A.dot(A)

        generator = FullBatchNodeGenerator(G, "test", transform=func)
        assert generator.name == "test"

        A = G.to_adjacency_matrix().toarray()
        assert np.array_equal(A.dot(A), generator.Aadj.toarray())

    def test_generator_methods(self):
        node_ids = list(self.G.nodes())
        Aadj = self.G.to_adjacency_matrix().toarray()
        Aadj_selfloops = Aadj + np.eye(*Aadj.shape) - np.diag(Aadj.diagonal())
        Dtilde = np.diag(Aadj_selfloops.sum(axis=1) ** (-0.5))
        Agcn = Dtilde.dot(Aadj_selfloops).dot(Dtilde)
        Appnp = 0.1 * np.linalg.inv(np.eye(Agcn.shape[0]) - ((1 - 0.1) * Agcn))

        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=True, method="none"
        )
        assert np.allclose(A_dense, Aadj)
        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=False, method="none"
        )
        assert np.allclose(A_dense, Aadj)

        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=True, method="self_loops"
        )
        assert np.allclose(A_dense, Aadj_selfloops)
        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=False, method="self_loops"
        )
        assert np.allclose(A_dense, Aadj_selfloops)

        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=True, method="gat"
        )
        assert np.allclose(A_dense, Aadj_selfloops)
        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=False, method="gat"
        )
        assert np.allclose(A_dense, Aadj_selfloops)

        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=True, method="gcn"
        )
        assert np.allclose(A_dense, Agcn)
        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=False, method="gcn"
        )
        assert np.allclose(A_dense, Agcn)

        # Check other pre-processing options
        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=True, method="sgc", k=2
        )
        assert np.allclose(A_dense, Agcn.dot(Agcn))
        A_dense, _, _ = self.generator_flow(
            self.G, node_ids, None, sparse=False, method="sgc", k=2
        )
        assert np.allclose(A_dense, Agcn.dot(Agcn))

        A_dense, _, _ = self.generator_flow(
            self.G,
            node_ids,
            None,
            sparse=False,
            method="ppnp",
            teleport_probability=0.1,
        )
        assert np.allclose(A_dense, Appnp)

        ppnp_sparse_failed = False
        try:
            A_dense, _, _ = self.generator_flow(
                self.G,
                node_ids,
                None,
                sparse=True,
                method="ppnp",
                teleport_probability=0.1,
            )
        except ValueError as e:
            ppnp_sparse_failed = True

        assert ppnp_sparse_failed


@pytest.mark.parametrize("sparse", [True, False])
def test_corrupt_full_batch_generator(sparse):

    G = example_graph_random(n_nodes=20)

    generator = FullBatchNodeGenerator(G, sparse=sparse)

    base_gen = generator.flow(G.nodes())
    gen = CorruptedNodeSequence(base_gen)

    [shuffled_feats, features, *_], targets = gen[0]

    assert features.shape == shuffled_feats.shape

    # check shuffled_feats are feats
    assert not np.array_equal(features, shuffled_feats)

    # check that all feature vecs in shuffled_feats correspond to a feature vec in features
    assert all(
        any(
            np.array_equal(shuffled_feats[:, i, :], features[:, j, :])
            for j in range(features.shape[1])
        )
        for i in range(shuffled_feats.shape[1])
    )


@pytest.mark.parametrize("is_directed", [True, False])
def test_corrupt_graphsage_generator(is_directed):

    G = example_graph_random(n_nodes=20, is_directed=is_directed)

    if is_directed:
        generator = DirectedGraphSAGENodeGenerator(
            G, batch_size=5, in_samples=[2, 3], out_samples=[4, 1]
        )
    else:
        generator = GraphSAGENodeGenerator(G, batch_size=5, num_samples=[2, 3])

    base_gen = generator.flow(G.nodes())
    gen = CorruptedNodeSequence(base_gen)

    x, targets = gen[0]
    clean_feats, _ = base_gen[0]

    shuffled_feats = x[: (len(x) // 2)]
    features = x[(len(x) // 2) :]

    assert len(clean_feats) == len(features)
    assert len(x) == 2 * len(clean_feats)
    assert len(features) == len(shuffled_feats)
    assert all(f.shape == s.shape for f, s in zip(features, shuffled_feats))

    features = np.concatenate(features, axis=1).reshape(-1, features[0][0].shape[-1])
    shuffled_feats = np.concatenate(shuffled_feats, axis=1).reshape(
        -1, features.shape[-1]
    )

    # check shuffled_feats are feats
    assert not np.array_equal(features, shuffled_feats)

    # check that all feature vecs in shuffled_feats correspond to a feature vec in features
    assert all(
        any(
            np.array_equal(shuffled_feats[i, :], features[j, :])
            for j in range(features.shape[0])
        )
        for i in range(shuffled_feats.shape[0])
    )
