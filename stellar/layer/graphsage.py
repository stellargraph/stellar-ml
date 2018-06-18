# -*- coding: utf-8 -*-
#
# Copyright 2018 Data61, CSIRO
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
GraphSAGE and compatible aggregator layers

"""


from keras.engine.topology import Layer
from keras import backend as K
from keras.layers import Lambda, Dropout, Reshape
from typing import List, Callable


class MeanAggregator(Layer):
    """
    Mean Aggregator for GraphSAGE implemented with Keras base layer

    """

    def __init__(self, output_dim: int, bias: bool = False, act: Callable = K.relu, **kwargs):
        """
        Construct mean aggregator

        :param output_dim:  Output dimension
        :param bias:        Optional bias
        :param act:         Activation function
        """

        self.output_dim = output_dim
        assert output_dim % 2 == 0
        self.half_output_dim = int(output_dim / 2)
        self.has_bias = bias
        self.act = act
        self.w_neigh = None
        self.w_self = None
        self.bias = None
        self._initializer = 'glorot_uniform'
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.w_neigh = self.add_weight(
            name='w_neigh',
            shape=(input_shape[1][3], self.half_output_dim),
            initializer=self._initializer,
            trainable=True
        )
        self.w_self = self.add_weight(
            name='w_self',
            shape=(input_shape[0][2], self.half_output_dim),
            initializer=self._initializer,
            trainable=True
        )
        if self.has_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=[self.output_dim],
                initializer='zeros',
                trainable=True
            )
        super().build(input_shape)

    def call(self, x, **kwargs):
        neigh_means = K.mean(x[1], axis=2)

        from_self = K.dot(x[0], self.w_self)
        from_neigh = K.dot(neigh_means, self.w_neigh)
        total = K.concatenate([from_self, from_neigh], axis=2)

        return self.act(total + self.bias if self.has_bias else total)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.output_dim


class Graphsage:
    """
    Implementation of the GraphSAGE algorithm with Keras layers.

    """

    def __init__(
            self,
            output_dims: List[int],
            n_samples: List[int],
            input_dim: int,
            aggregator: Layer = MeanAggregator,
            bias: bool = False,
            dropout: float = 0.
    ):
        """
        Construct aggregator and other supporting layers for GraphSAGE

        :param output_dims: Output dimension at each layer
        :param n_samples:   Number of neighbours sampled for each hop/layer
        :param input_dim:   Feature vector dimension
        :param aggregator:  Aggregator class
        :param bias:        Optional bias
        :param dropout:     Optional dropout
        """

        assert len(n_samples) == len(output_dims)
        self.n_layers = len(n_samples)
        self.n_samples = n_samples
        self.dims = [input_dim] + output_dims
        self.bias = bias
        self._dropout = Dropout(dropout)
        self._aggs = [aggregator(self.dims[layer+1],
                                 bias=self.bias,
                                 act=K.relu if layer < self.n_layers - 1 else lambda x: x)
                      for layer in range(self.n_layers)]
        self._neigh_reshape = [[Reshape((-1, self.n_samples[i], self.dims[layer]))
                                for i in range(self.n_layers - layer)]
                               for layer in range(self.n_layers)]
        self._normalization = Lambda(lambda x: K.l2_normalize(x, 2))

    def __call__(self, x: List):
        """
        Apply aggregator layers

        :param x:       Batch input features
        :return:        Output tensor
        """

        def compose_layers(x: List, layer: int):
            """
            Function to recursively compose aggregation layers. When current layer is at final layer, then length of x
            should be 1, and compose_layers(x, layer) returns x[0].

            :param x:       List of feature matrix tensors
            :param layer:   Current layer index
            :return:        x computed from current layer to output layer
            """

            def x_next(agg):
                return [agg([self._dropout(x[i]),
                             self._dropout(self._neigh_reshape[layer][i](x[i+1]))])
                        for i in range(self.n_layers - layer)]

            return compose_layers(x_next(self._aggs[layer]), layer + 1) if layer < self.n_layers else x

        return self._normalization(compose_layers(x, 0))
