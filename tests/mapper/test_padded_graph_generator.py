# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
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
from stellargraph.core.graph import *
from stellargraph.mapper.padded_graph_generator import (
    PaddedGraphGenerator,
    PaddedGraphSequence,
)

import numpy as np
import pytest
from ..test_utils.graphs import example_graph_random, example_graph, example_hin_1

graphs = [
    example_graph_random(feature_size=4, n_nodes=6),
    example_graph_random(feature_size=4, n_nodes=5),
    example_graph_random(feature_size=4, n_nodes=3),
    example_graph_random(feature_size=4, node_types=0, edge_types=0),
]


def test_generator_init():
    generator = PaddedGraphGenerator(graphs=graphs)
    assert len(generator.graphs) == len(graphs)


def test_generator_init_different_feature_numbers():
    graphs_diff_num_features = [
        example_graph_random(feature_size=2, n_nodes=6),
        example_graph_random(feature_size=4, n_nodes=5),
    ]

    with pytest.raises(
        ValueError,
        match="graphs: expected node features for all graph to have same dimensions,.*2.*4",
    ):
        generator = PaddedGraphGenerator(graphs=graphs_diff_num_features)


def test_generator_init_nx_graph():
    graphs_nx = [
        example_graph_random(feature_size=4, n_nodes=3, is_directed=False),
        example_graph_random(
            feature_size=4, n_nodes=2, is_directed=False
        ).to_networkx(),
    ]

    with pytest.raises(
        TypeError, match="graphs: expected.*StellarGraph.*found MultiGraph."
    ):
        generator = PaddedGraphGenerator(graphs=graphs_nx)


def test_generator_init_hin():
    graphs_mixed = [
        example_graph_random(feature_size=2, n_nodes=6),
        example_hin_1(is_directed=False),
    ]

    with pytest.raises(
        ValueError,
        match="graphs: expected only graphs with a single node type.*found.*'A', 'B'",
    ):
        generator = PaddedGraphGenerator(graphs=graphs_mixed)


def test_generator_flow_invalid_batch_size():
    with pytest.raises(
        ValueError, match="expected batch_size.*strictly positive integer, found -1"
    ):
        PaddedGraphGenerator(graphs=graphs).flow(graph_ilocs=[0], batch_size=-1)

    with pytest.raises(
        TypeError, match="expected batch_size.*integer type, found float"
    ):
        PaddedGraphGenerator(graphs=graphs).flow(graph_ilocs=[0], batch_size=2.0)

    with pytest.raises(
        ValueError, match="expected batch_size.*strictly positive integer, found 0"
    ):
        PaddedGraphGenerator(graphs=graphs).flow(graph_ilocs=[0], batch_size=0)


def test_generator_flow_incorrect_targets():

    generator = PaddedGraphGenerator(graphs=graphs)

    with pytest.raises(
        ValueError, match="expected targets to be the same length as node_ids,.*1 vs 2"
    ):
        generator.flow(graph_ilocs=[0, 1], targets=np.array([0]))

    with pytest.raises(
        TypeError, match="targets: expected an iterable or None object, found int"
    ):
        generator.flow(graph_ilocs=[0, 1], targets=1)


def test_generator_flow_no_targets():

    generator = PaddedGraphGenerator(graphs=graphs)

    seq = generator.flow(graph_ilocs=[0, 1, 2], batch_size=2)
    assert isinstance(seq, PaddedGraphSequence)

    assert len(seq) == 2  # two batches

    # The first batch should be size 2 and the second batch size 1
    batch_0 = seq[0]
    assert batch_0[0][0].shape[0] == 2
    assert batch_0[0][1].shape[0] == 2
    assert batch_0[0][2].shape[0] == 2
    assert batch_0[1] is None

    batch_1 = seq[1]
    assert batch_1[0][0].shape[0] == 1
    assert batch_1[0][1].shape[0] == 1
    assert batch_1[0][2].shape[0] == 1
    assert batch_1[1] is None


def test_generator_flow_check_padding():

    generator = PaddedGraphGenerator(graphs=graphs)

    seq = generator.flow(graph_ilocs=[0, 2], batch_size=2)
    assert isinstance(seq, PaddedGraphSequence)

    assert len(seq) == 1

    # The largest graph has 6 nodes vs 3 for the smallest one.
    # Check that the data matrices have the correct size 6
    batch = seq[0]

    assert batch[0][0].shape == (2, 6, 4)
    assert batch[0][1].shape == (2, 6)
    assert batch[0][2].shape == (2, 6, 6)

    for mask in batch[0][1]:
        assert np.sum(mask) == 6 or np.sum(mask) == 3


def test_generator_flow_with_targets():

    generator = PaddedGraphGenerator(graphs=graphs)

    seq = generator.flow(graph_ilocs=[1, 2], targets=np.array([0, 1]), batch_size=1)
    assert isinstance(seq, PaddedGraphSequence)

    for batch in seq:
        assert batch[0][0].shape[0] == 1
        assert batch[0][1].shape[0] == 1
        assert batch[0][2].shape[0] == 1
        assert batch[1].shape[0] == 1


@pytest.mark.parametrize("symmetric_normalization", [True, False])
def test_generator_adj_normalisation(symmetric_normalization):

    graph = example_graph(feature_size=4)

    generator = PaddedGraphGenerator(graphs=[graph])
    seq = generator.flow(
        graph_ilocs=[0], symmetric_normalization=symmetric_normalization
    )

    adj_norm_seq = seq.normalized_adjs[0].todense()

    adj = np.array(graph.to_adjacency_matrix().todense())
    np.fill_diagonal(adj, 1)
    if symmetric_normalization:
        inv_deg = np.diag(np.sqrt(1.0 / adj.sum(axis=1)))
        adj_norm = inv_deg.dot(adj).dot(inv_deg)
    else:
        inv_deg = np.diag(1.0 / adj.sum(axis=1))
        adj_norm = inv_deg.dot(adj)

    assert np.allclose(adj_norm_seq, adj_norm)


def test_generator_flow_shuffle():

    generator = PaddedGraphGenerator(graphs=graphs)
    num_epochs_to_check = 5

    def get_batches(seq):
        return [seq[i][0] for i in range(len(seq))]

    def batches_all_equal(batches, other_batches):
        checks = [
            inp.shape == other_inp.shape and np.allclose(inp, other_inp)
            for batch, other_batch in zip(batches, other_batches)
            for inp, other_inp in zip(batch, other_batch)
        ]
        return all(checks)

    def get_next_epoch_batches(seq):
        seq.on_epoch_end()
        return get_batches(seq)

    # shuffle = False
    seq = generator.flow(graph_ilocs=[0, 1, 2], batch_size=2, shuffle=False)
    batches = get_batches(seq)
    for _ in range(num_epochs_to_check):
        assert batches_all_equal(batches, get_next_epoch_batches(seq))

    # shuffle = True, fixed seed
    seq = generator.flow(graph_ilocs=[0, 1, 2], batch_size=2, shuffle=True, seed=0)
    batches = get_batches(seq)
    at_least_one_different = False
    for _ in range(num_epochs_to_check):
        if not batches_all_equal(batches, get_next_epoch_batches(seq)):
            at_least_one_different = True
    assert at_least_one_different


def test_generator_empty_graph():
    generator = PaddedGraphGenerator(graphs=graphs)
    seq_alone = generator.flow([3], batch_size=1)
    assert len(seq_alone) == 1
    (feats, masks, adj_graphs), _ = seq_alone[0]
    assert feats.shape == (1, 0, 0)
    assert masks.shape == (1, 0)
    assert adj_graphs.shape == (1, 0, 0)

    seq_together = generator.flow([2, 3], batch_size=2, shuffle=False)
    assert len(seq_together) == 1
    (feats, masks, adj_graphs), _ = seq_together[0]
    assert feats.shape == (2, 3, 4)
    assert masks.shape == (2, 3)
    assert adj_graphs.shape == (2, 3, 3)
    np.testing.assert_array_equal(masks, [[True] * 3, [False] * 3])
    np.testing.assert_array_equal(adj_graphs[1, ...], [[0] * 3] * 3)
