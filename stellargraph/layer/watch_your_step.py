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

import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Input, Lambda, Concatenate, Dense
from tensorflow.keras import backend as K
import numpy as np
import warnings
from ..mapper.adjacency_generators import AdjacencyPowerGenerator
from ..core.experimental import experimental


class AttentiveWalk(Layer):
    """
    This implements the graph attention as in Watch Your Step: Learning Node Embeddings via Graph Attention
    https://arxiv.org/pdf/1710.09599.pdf.

    Args:
        walk_length (int): the length of the random walks. Equivalent to the number of adjacency powers used. Defaults
            to `10` as this value was found to perform well by the authors of the paper.
        attention_initializer (str or func, optional): The initialiser to use for the attention weights.
        attention_regularizer (str or func, optional): The regulariser to use for the attention weights.
        attention_constraint (str or func, optional): The constraint to use for the attention weights.
        input_dim (tuple of ints, optional): The shape of the input to the layer.
    """

    def __init__(
        self,
        walk_length=10,
        attention_initializer="glorot_uniform",
        attention_regularizer=None,
        attention_constraint=None,
        input_dim=None,
        **kwargs,
    ):

        if "input_shape" not in kwargs and input_dim is not None:
            kwargs["input_shape"] = input_dim

        self.walk_length = walk_length
        self.attention_initializer = attention_initializer
        self.attention_regularizer = attention_regularizer
        self.attention_constraint = attention_constraint
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][-1],)

    compute_output_shape.__doc__ = Layer.compute_output_shape.__doc__

    def build(self, input_shapes):

        self.attention_weights = self.add_weight(
            shape=(self.walk_length,),
            initializer=self.attention_initializer,
            name="attention_weights",
            regularizer=self.attention_regularizer,
            constraint=self.attention_constraint,
        )

        self.built = True

    build.__doc__ = Layer.build.__doc__

    def call(self, partial_powers):
        """
        Applies the layer and calculates the expected random walks.

        Args:
            partial_powers: num_rows rows of the first num_powers powers of adjacency matrix with shape
            (num_rows, num_powers, num_nodes)

        Returns:
            Tensor that represents the expected random walks starting from nodes corresponding to the input rows of
            shape (num_rows, num_nodes)
        """

        attention = K.softmax(self.attention_weights)
        expected_walk = tf.einsum("ijk,j->ik", partial_powers, attention)

        return expected_walk


@experimental(reason="lack of unit tests")
def get_embeddings(model):
    """
    This function returns the embeddings from a model with Watch Your Step embeddings.

    Args:
        model (keras Model): a keras model that contains Watch Your Step embeddings.

    Returns:
        embeddings (np.array): a numpy array of the model's embeddings.
    """
    embeddings = np.hstack(
        [
            model.get_layer("WATCH_YOUR_STEP_LEFT_EMBEDDINGS").embeddings.numpy(),
            model.get_layer("WATCH_YOUR_STEP_RIGHT_EMBEDDINGS")
            .kernel.numpy()
            .transpose(),
        ]
    )

    return embeddings


@experimental(reason="lack of unit tests")
class WatchYourStep:
    """
    Implementation of the node embeddings as in Watch Your Step: Learning Node Embeddings via Graph Attention
    https://arxiv.org/pdf/1710.09599.pdf.

    This model requires specification of the number of random walks starting from each node, and the embedding dimension
    to use for the node embeddings.

    Note:
        - the embedding dimension should be an even number or else will be rounded down to nearest even number.

    Args:
        generator (AdjacencyPowerGenerator): the generator
        num_walks (int): the number of random walks starting at each node to use when calculating the expected random
            walks. Defaults to `80` as this value was found to perform well by the authors of the paper.
        embedding dimension (int): the dimension to use for the node embeddings. Defaults to `64`
            as this value was found to perform well by the authors of the paper.
        attention_initializer (str or func, optional): The initialiser to use for the attention weights.
        attention_regularizer (str or func, optional): The regulariser to use for the attention weights.
        attention_constraint (str or func, optional): The constraint to use for the attention weights.
    """

    def __init__(
        self,
        generator,
        num_walks=80,
        embedding_dimension=64,
        attention_initializer="glorot_uniform",
        attention_regularizer=None,
        attention_constraint=None,
    ):

        if not isinstance(generator, AdjacencyPowerGenerator):
            raise TypeError(
                "generator should be an instance of AdjacencyPowerGenerator."
            )

        if not isinstance(num_walks, int):
            raise TypeError("num_walks should be an int.")

        if num_walks <= 0:
            raise ValueError("num_walks should be a positive int.")

        self.num_walks = num_walks
        self.num_powers = generator.num_powers
        self.n_nodes = int(generator.Aadj_T.shape[0])

        if embedding_dimension % 2 != 0:
            warnings.warn(
                f"embedding_dimension: expected even number, found odd number ({embedding_dimension}). It will be rounded down to {embedding_dimension - 1}."
            )
            embedding_dimension -= 1

        self.embedding_dimension = embedding_dimension
        self.attention_regularizer = attention_regularizer
        self.attention_initializer = attention_initializer
        self.attention_constraint = attention_constraint

    def build(self):
        """
        This function builds the layers for a keras model.

        returns:
            A tuple of (inputs, outputs) to use with a keras model.
        """

        input_rows = Input(batch_shape=(None,), name="row_node_ids", dtype="int64")
        input_powers = Input(batch_shape=(None, self.num_powers, self.n_nodes))

        left_embedding = Embedding(
            self.n_nodes,
            int(self.embedding_dimension / 2),
            input_length=None,
            name="WATCH_YOUR_STEP_LEFT_EMBEDDINGS",
        )

        vectors_left = left_embedding(input_rows)

        # all right embeddings are used in every batch. to avoid unnecessary lookups the right embeddings are stored
        # in a dense layer to enable efficient dot product between the left vectors in the current batch and all right
        # vectors
        outer_product = Dense(
            self.n_nodes,
            use_bias=False,
            kernel_initializer="uniform",
            name="WATCH_YOUR_STEP_RIGHT_EMBEDDINGS",
        )(vectors_left)

        sigmoids = tf.keras.activations.sigmoid(outer_product)
        attentive_walk_layer = AttentiveWalk(
            walk_length=self.num_powers,
            attention_constraint=self.attention_constraint,
            attention_regularizer=self.attention_regularizer,
            attention_initializer=self.attention_initializer,
        )
        expected_walk = self.num_walks * attentive_walk_layer(input_powers)

        # layer to add batch dimension of 1 to output
        expander = Lambda(lambda x: K.expand_dims(x, axis=1))

        output = Concatenate(axis=1)([expander(expected_walk), expander(sigmoids)])

        return [input_rows, input_powers], output
