# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
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
The layer package contains implementations of popular neural network layers for graph ML as Keras layers
"""

# __all__ = ["graphsage", "hinsage", "link_inference"]

# Expose the layers
from .graphsage import *
from .hinsage import *
from .graph_attention import *
from .link_inference import *
from .ppnp import *
from .appnp import *
from .gcn import *
from .cluster_gcn import *
from .attri2vec import *
from .node2vec import *
from .misc import SqueezedSparseConversion
from .preprocessing_layer import GraphPreProcessingLayer
from .rgcn import *
from .watch_your_step import *
from .knowledge_graph import *
from .graph_classification import *
from .deep_graph_infomax import *
from .sort_pooling import *
from .gcn_lstm import *
from .ugraphemb_attention_pooling import UGraphEmbAttentionPooling
