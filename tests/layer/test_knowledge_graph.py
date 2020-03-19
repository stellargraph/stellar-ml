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

import itertools

import pytest

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, initializers, losses

from stellargraph import StellarGraph, StellarDiGraph
from stellargraph.mapper.knowledge_graph import KGTripleGenerator
from stellargraph.layer.knowledge_graph import ComplEx, DistMult

from .. import test_utils
from ..test_utils.graphs import knowledge_graph


pytestmark = [
    test_utils.ignore_stellargraph_experimental_mark,
    pytest.mark.filterwarnings(
        r"ignore:ComplEx:stellargraph.core.experimental.ExperimentalWarning"
    ),
]


def triple_df(*values):
    return pd.DataFrame(values, columns=["source", "label", "target"])


def test_complex(knowledge_graph):
    # this test creates a random untrained model and predicts every possible edge in the graph, and
    # compares that to a direct implementation of the scoring method in the paper
    gen = KGTripleGenerator(knowledge_graph, 3)

    # use a random initializer with a large positive range, so that any differences are obvious
    init = initializers.RandomUniform(-1, 1)
    complex_model = ComplEx(gen, 5, embeddings_initializer=init)
    x_inp, x_out = complex_model.build()

    model = Model(x_inp, x_out)
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True))

    every_edge = itertools.product(
        knowledge_graph.nodes(),
        knowledge_graph._edges.types.pandas_index,
        knowledge_graph.nodes(),
    )
    df = triple_df(*every_edge)

    # check the model can be trained on a few (uneven) batches
    model.fit(
        gen.flow(df.iloc[:7], negative_samples=2),
        validation_data=gen.flow(df.iloc[7:14], negative_samples=3),
    )

    # compute the exact values based on the model by extracting the embeddings for each element and
    # doing the Re(<e_s, w_r, conj(e_o)>) inner product
    s_idx = knowledge_graph._get_index_for_nodes(df.source)
    r_idx = knowledge_graph._edges.types.to_iloc(df.label)
    o_idx = knowledge_graph._get_index_for_nodes(df.target)

    nodes, edge_types = ComplEx.embeddings(model)
    # the rows correspond to the embeddings for the given edge, so we can do bulk operations
    e_s = nodes[s_idx, :]
    w_r = edge_types[r_idx, :]
    e_o = nodes[o_idx, :]
    actual = (e_s * w_r * e_o.conj()).sum(axis=1).real

    # predict every edge using the model
    prediction = model.predict(gen.flow(df))

    # (use an absolute tolerance to allow for catastrophic cancellation around very small values)
    assert np.allclose(prediction[:, 0], actual, rtol=1e-3, atol=1e-14)

    # the model is stateful (i.e. it holds the weights permanently) so the embeddings with a second
    # 'build' should be the same as the original one
    model2 = Model(*complex_model.build())
    nodes2, edge_types2 = ComplEx.embeddings(model2)
    assert np.array_equal(nodes, nodes2)
    assert np.array_equal(edge_types, edge_types2)


def test_complex_rankings():
    nodes = pd.DataFrame(index=["a", "b", "c", "d"])
    rels = ["W", "X", "Y", "Z"]
    empty = pd.DataFrame(columns=["source", "target"])

    every_edge = itertools.product(nodes.index, rels, nodes.index)
    df = triple_df(*every_edge)

    no_edges = StellarDiGraph(nodes, {name: empty for name in rels})

    # the filtering is most interesting when there's a smattering of edges, somewhere between none
    # and all; this does a stratified sample by label, to make sure there's at least one edge from
    # each label.
    one_per_label_df = df.groupby("label").apply(lambda df: df.sample(n=1)).droplevel(0)
    others_df = df.sample(frac=0.25)
    some_edges_df = pd.concat([one_per_label_df, others_df], ignore_index=True)

    some_edges = StellarDiGraph(
        nodes,
        {name: df.drop(columns="label") for name, df in some_edges_df.groupby("label")},
    )

    all_edges = StellarDiGraph(
        nodes=nodes,
        edges={name: df.drop(columns="label") for name, df in df.groupby("label")},
    )

    gen = KGTripleGenerator(all_edges, 3)
    x_inp, x_out = ComplEx(gen, 5).build()
    model = Model(x_inp, x_out)

    raw_some, filtered_some = ComplEx.rank_edges_against_all_nodes(
        model, gen.flow(df), some_edges
    )
    # basic check that the ranks are formed correctly
    assert raw_some.dtype == int
    assert np.all(raw_some >= 1)
    # filtered ranks are never greater, and sometimes less
    assert np.all(filtered_some <= raw_some)
    assert np.any(filtered_some < raw_some)

    raw_no, filtered_no = ComplEx.rank_edges_against_all_nodes(
        model, gen.flow(df), no_edges
    )
    np.testing.assert_array_equal(raw_no, raw_some)
    # with no edges, filtering does nothing
    np.testing.assert_array_equal(raw_no, filtered_no)

    raw_all, filtered_all = ComplEx.rank_edges_against_all_nodes(
        model, gen.flow(df), all_edges
    )
    np.testing.assert_array_equal(raw_all, raw_some)
    # when every edge is known, the filtering should eliminate every possibility
    assert np.all(filtered_all == 1)

    # check the ranks against computing them from the model predictions directly. That is, for each
    # edge, compare the rank against one computed by counting the predictions. This computes the
    # filtered ranks naively too.
    predictions = model.predict(gen.flow(df))

    for (source, rel, target), score, raw, filtered in zip(
        df.itertuples(index=False), predictions, raw_some, filtered_some
    ):
        # rank for the subset specified by the given selector
        def rank(compare_selector):
            return 1 + (predictions[compare_selector] > score).sum()

        same_r = df.label == rel

        same_s_r = (df.source == source) & same_r

        expected_raw_mod_o_rank = rank(same_s_r)
        assert raw[0] == expected_raw_mod_o_rank

        known_objects = some_edges_df[
            (some_edges_df.source == source) & (some_edges_df.label == rel)
        ]
        object_is_unknown = ~df.target.isin(known_objects.target)
        expected_filt_mod_o_rank = rank(same_s_r & object_is_unknown)
        assert filtered[0] == expected_filt_mod_o_rank

        same_r_o = same_r & (df.target == target)

        expected_raw_mod_s_rank = rank(same_r_o)
        assert raw[1] == expected_raw_mod_s_rank

        known_subjects = some_edges_df[
            (some_edges_df.label == rel) & (some_edges_df.target == target)
        ]
        subject_is_unknown = ~df.source.isin(known_subjects.source)
        expected_filt_mod_s_rank = rank(subject_is_unknown & same_r_o)
        assert filtered[1] == expected_filt_mod_s_rank


def test_dismult(knowledge_graph):
    # this test creates a random untrained model and predicts every possible edge in the graph, and
    # compares that to a direct implementation of the scoring method in the paper
    gen = KGTripleGenerator(knowledge_graph, 3)

    # use a random initializer with a large range, so that any differences are obvious
    init = initializers.RandomUniform(-1, 1)
    distmult_model = DistMult(gen, 5, embeddings_initializer=init)
    x_inp, x_out = distmult_model.build()

    model = Model(x_inp, x_out)

    model.compile(loss=losses.BinaryCrossentropy(from_logits=True))

    every_edge = itertools.product(
        knowledge_graph.nodes(),
        knowledge_graph._edges.types.pandas_index,
        knowledge_graph.nodes(),
    )
    df = triple_df(*every_edge)

    # check the model can be trained on a few (uneven) batches
    model.fit(
        gen.flow(df.iloc[:7], negative_samples=2),
        validation_data=gen.flow(df.iloc[7:14], negative_samples=3),
    )

    # compute the exact values based on the model by extracting the embeddings for each element and
    # doing the y_(e_1)^T M_r y_(e_2) = <e_1, w_r, e_2> inner product
    s_idx = knowledge_graph._get_index_for_nodes(df.source)
    r_idx = knowledge_graph._edges.types.to_iloc(df.label)
    o_idx = knowledge_graph._get_index_for_nodes(df.target)

    nodes, edge_types = DistMult.embeddings(model)
    # the rows correspond to the embeddings for the given edge, so we can do bulk operations
    e_s = nodes[s_idx, :]
    w_r = edge_types[r_idx, :]
    e_o = nodes[o_idx, :]
    actual = (e_s * w_r * e_o).sum(axis=1)

    # predict every edge using the model
    prediction = model.predict(gen.flow(df))

    # (use an absolute tolerance to allow for catastrophic cancellation around very small values)
    assert np.allclose(prediction[:, 0], actual, rtol=1e-3, atol=1e-14)

    # the model is stateful (i.e. it holds the weights permanently) so the embeddings with a second
    # 'build' should be the same as the original one
    model2 = Model(*distmult_model.build())
    nodes2, edge_types2 = DistMult.embeddings(model2)
    assert np.array_equal(nodes, nodes2)
    assert np.array_equal(edge_types, edge_types2)
