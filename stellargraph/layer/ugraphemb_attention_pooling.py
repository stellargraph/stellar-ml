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

__all__ = ["UGraphEmbAttentionPooling"]

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations, initializers, constraints, regularizers, backend as K
from ..core.experimental import experimental
from ..core.validation import require_integer_in_range

class UGraphEmbAttentionPooling(Layer):
    def __init__(
        self,
        attention_activation="relu",
        attention_initializer="glorot_uniform",
        attention_regularizer=None,
        attention_constraint=None,
        **kwargs,
    ):
        self.attention_activation = activations.get(attention_activation)
        self.attention_initializer = initializers.get(attention_initializer)
        self.attention_regularizer = regularizers.get(attention_regularizer)
        self.attention_constraint = constraints.get(attention_constraint)

        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            attention_activation=activations.serialize(self.attention_activation),
            attention_initializer=initializers.serialize(self.attention_initializer),
            attention_regularizer=regularizers.serialize(self.attention_regularizer),
            attention_constraint=constraints.serialize(self.attention_constraint),
        )
        return config

    def build(self, input_shape):
        batch_size, num_nodes, dimension = input_shape

        self.attention_kernel = self.add_weight(
            shape=(dimension, dimension),
            name="attention_kernel",
            initializer=self.attention_initializer,
            regularizer=self.attention_regularizer,
            constraint=self.attention_constraint,
        )

        super().build(input_shape)

    def call(self, embeddings, mask):
        # embeddings.shape == (B, N, D), mask.shape == (B, N)

        # shape == (B, N, 1)
        element_mask = tf.cast(mask, embeddings.dtype)[..., None]

        # shape == (B, D)
        graph_means = tf.reduce_sum(embeddings * element_mask, axis=1) / tf.reduce_sum(element_mask, axis=1)

        # shape == (B, D)
        query = self.attention_activation(K.dot(graph_means, self.attention_kernel))

        # shape == (B, N)
        weights = K.sigmoid(K.batch_dot(embeddings, query))

        # shape == (B, N, D)
        weighted = weights[..., None] * embeddings * element_mask

        # shape == (B, D)
        return tf.reduce_sum(weighted, axis=1)
