![StellarGraph Machine Learning library logo](https://raw.githubusercontent.com/stellargraph/stellargraph/develop/stellar-graph-banner.png)

<p align="center">
  <a href="https://stellargraph.readthedocs.io/" alt="Docs"><img src="https://readthedocs.org/projects/stellargraph/badge/?version=latest"/></a>
  <a href="https://github.com/stellargraph/stellargraph/discussions" alt="Discussions forum"><img src="https://img.shields.io/badge/help_forum-discussions-blue.svg"/></a>
  <a href="https://pypi.org/project/stellargraph/" alt="PyPI"><img src="https://img.shields.io/pypi/v/stellargraph.svg"/></a>
  <a href="https://github.com/stellargraph/stellargraph/blob/develop/LICENSE" alt="license"><img src="https://img.shields.io/github/license/stellargraph/stellargraph.svg"/></a>
</p>
<p align="center">
  <a href="https://github.com/stellargraph/stellargraph/blob/develop/CONTRIBUTING.md" alt="contributions welcome"><img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg"/></a>
  <a href="https://github.com/stellargraph/stellargraph/actions?query=branch%3Amaster" alt="Build status: master"><img src="https://github.com/stellargraph/stellargraph/workflows/CI/badge.svg?branch=master"/></a>
  <a href="https://github.com/stellargraph/stellargraph/actions?query=branch%3Adevelop" alt="Build status: develop"><img src="https://github.com/stellargraph/stellargraph/workflows/CI/badge.svg?branch=develop"/></a>
  <a href="https://codecov.io/gh/stellargraph/stellargraph"><img src="https://codecov.io/gh/stellargraph/stellargraph/branch/develop/graph/badge.svg" /></a>
  <a href="https://pypi.org/project/stellargraph" alt="pypi downloads"><img alt="pypi downloads" src="https://pepy.tech/badge/stellargraph"></a>
</p>


# StellarGraph Machine Learning Library

**StellarGraph** is a Python library for machine learning on [graphs and networks](https://en.wikipedia.org/wiki/Graph_%28discrete_mathematics%29).

## Table of Contents
   * [Introduction](#introduction)
   * [Getting Started](#getting-started)
   * [Getting Help](#getting-help)
   * [Example: GCN](#example-gcn)
   * [Algorithms](#algorithms)
   * [Installation](#installation)
       * [Install StellarGraph using PyPI](#install-stellargraph-using-pypi)
       * [Install StellarGraph in Anaconda Python](#install-stellargraph-in-anaconda-python)
       * [Install StellarGraph from GitHub source](#install-stellargraph-from-github-source)
   * [Citing](#citing)
   * [References](#references)

## Introduction

The StellarGraph library offers state-of-the-art algorithms for [graph machine learning](https://medium.com/stellargraph/knowing-your-neighbours-machine-learning-on-graphs-9b7c3d0d5896), making it easy to discover patterns and answer questions about graph-structured data. It can solve many machine learning tasks:

- Representation learning for nodes and edges, to be used for visualisation and various downstream machine learning tasks;
- [Classification and attribute inference of nodes](https://medium.com/stellargraph/can-graph-machine-learning-identify-hate-speech-in-online-social-networks-58e3b80c9f7e) or edges;
- Classification of whole graphs;
- Link prediction;
- [Interpretation of node classification](https://medium.com/stellargraph/https-medium-com-stellargraph-saliency-maps-for-graph-machine-learning-5cca536974da) [8].

Graph-structured data represent entities as nodes (or vertices) and relationships between them as edges (or links), and can include data associated with either as attributes. For example, a graph can contain people as nodes and friendships between them as links, with data like a person's age and the date a friendship was established. StellarGraph supports analysis of many kinds of graphs:

- homogeneous (with nodes and links of one type),
- heterogeneous (with more than one type of nodes and/or links)
- knowledge graphs (extreme heterogeneous graphs with thousands of types of edges)
- graphs with or without data associated with nodes
- graphs with edge weights

StellarGraph is built on [TensorFlow 2](https://tensorflow.org/) and its [Keras high-level API](https://www.tensorflow.org/guide/keras), as well as [Pandas](https://pandas.pydata.org) and [NumPy](https://www.numpy.org). It is thus user-friendly, modular and extensible. It interoperates smoothly with code that builds on these, such as the standard Keras layers and [scikit-learn](http://scikit-learn.github.io/stable), so it is easy to augment the core graph machine learning algorithms provided by StellarGraph. It is thus also [easy to install with `pip` or Anaconda](#installation).

## Getting Started

[The numerous detailed and narrated examples][demos] are a good way to get started with StellarGraph. There is likely to be one that is similar to your data or your problem (if not, [let us know](#getting-help)).

[demos]: https://stellargraph.readthedocs.io/en/stable/demos/index.html

You can start working with the examples immediately in Google Colab or Binder by clicking the ![](https://colab.research.google.com/assets/colab-badge.svg) and ![](https://mybinder.org/badge_logo.svg) badges within each Jupyter notebook.

Alternatively, you can run download a local copy of the demos and run them using `jupyter`. The demos can be downloaded by cloning the `master` branch of this repository, or by using the `curl` command below:
```bash
curl -L https://github.com/stellargraph/stellargraph/archive/master.zip | tar -xz --strip=1 stellargraph-master/demos
```

The dependencies required to run most of our demo notebooks locally can be installed using one of the following:

- Using `pip`: `pip install stellargraph[demos]`
- Using `conda`: `conda install -c stellargraph stellargraph`

(See [Installation](#installation) section for more details and more options.)

## Getting Help

If you get stuck or have a problem, there are many ways to make progress and get help or support:

- [Read the documentation](https://stellargraph.readthedocs.io)
- [Consult the examples][demos]
- Contact us:
  - [Ask questions and discuss problems on the StellarGraph Discussions forum](https://github.com/stellargraph/stellargraph/discussions)
  - [File an issue](https://github.com/stellargraph/stellargraph/issues/new/choose)
  - Send us an email at [stellargraph.io@gmail.com](mailto:stellargraph.io@gmail.com?subject=Question%20about%20the%20StellarGraph%20library)


## Example: GCN

One of the earliest deep machine learning algorithms for graphs is a Graph Convolution Network (GCN) [6]. The following example uses it for node classification: predicting the class from which a node comes. It shows how easy it is to apply using StellarGraph, and shows how StellarGraph integrates smoothly with Pandas and TensorFlow and libraries built on them.

#### Data preparation

Data for StellarGraph can be prepared using common libraries like Pandas and scikit-learn.

``` python
import pandas as pd
from sklearn import model_selection

def load_my_data():
    # your own code to load data into Pandas DataFrames, e.g. from CSV files or a database
    ...

nodes, edges, targets = load_my_data()

# Use scikit-learn to compute training and test sets
train_targets, test_targets = model_selection.train_test_split(targets, train_size=0.5)
```

#### Graph machine learning model

This is the only part that is specific to StellarGraph. The machine learning model consists of some graph convolution layers followed by a layer to compute the actual predictions as a TensorFlow tensor. StellarGraph makes it easy to construct all of these layers via the `GCN` model class. It also makes it easy to get input data in the right format via the `StellarGraph` graph data type and a data generator.

```python
import stellargraph as sg
import tensorflow as tf

# convert the raw data into StellarGraph's graph format for faster operations
graph = sg.StellarGraph(nodes, edges)

generator = sg.mapper.FullBatchNodeGenerator(graph, method="gcn")

# two layers of GCN, each with hidden dimension 16
gcn = sg.layer.GCN(layer_sizes=[16, 16], generator=generator)
x_inp, x_out = gcn.in_out_tensors() # create the input and output TensorFlow tensors

# use TensorFlow Keras to add a layer to compute the (one-hot) predictions
predictions = tf.keras.layers.Dense(units=len(ground_truth_targets.columns), activation="softmax")(x_out)

# use the input and output tensors to create a TensorFlow Keras model
model = tf.keras.Model(inputs=x_inp, outputs=predictions)
```

#### Training and evaluation

The model is a conventional TensorFlow Keras model, and so tasks such as training and evaluation can use the functions offered by Keras. StellarGraph's data generators make it simple to construct the required Keras Sequences for input data.

```python
# prepare the model for training with the Adam optimiser and an appropriate loss function
model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

# train the model on the train set
model.fit(generator.flow(train_targets.index, train_targets), epochs=5)

# check model generalisation on the test set
(loss, accuracy) = model.evaluate(generator.flow(test_targets.index, test_targets))
print(f"Test set: loss = {loss}, accuracy = {accuracy}")
```

This algorithm is spelled out in more detail in [its extended narrated notebook][gcn-demo]. We provide [many more algorithms, each with a detailed example][demos].

[gcn-demo]: https://stellargraph.readthedocs.io/en/stable/demos/node-classification/gcn-node-classification.html

## Algorithms
The StellarGraph library currently includes the following algorithms for graph machine learning:

| Algorithm | Description |
| --- | --- |
| GraphSAGE [1] | Supports supervised as well as unsupervised representation learning, node classification/regression, and link prediction for homogeneous networks. The current implementation supports multiple aggregation methods, including mean, maxpool, meanpool, and attentional aggregators. |
| HinSAGE | Extension of GraphSAGE algorithm to heterogeneous networks. Supports representation learning, node classification/regression, and link prediction/regression for heterogeneous graphs. The current implementation supports mean aggregation of neighbour nodes, taking into account their types and the types of links between them. |
| attri2vec [4] | Supports node representation learning, node classification, and out-of-sample node link prediction for homogeneous graphs with node attributes. |
| Graph ATtention Network (GAT) [5] | The GAT algorithm supports representation learning and node classification for homogeneous graphs. There are versions of the graph attention layer that support both sparse and dense adjacency matrices. |
| Graph Convolutional Network (GCN) [6] | The GCN algorithm supports representation learning and node classification for homogeneous graphs. There are versions of the graph convolutional layer that support both sparse and dense adjacency matrices. |
| Cluster Graph Convolutional Network (Cluster-GCN) [10] | An extension of the GCN algorithm supporting representation learning and node classification for homogeneous graphs. Cluster-GCN scales to larger graphs and can be used to train deeper GCN models using Stochastic Gradient Descent. |
| Simplified Graph Convolutional network (SGC) [7] | The SGC network algorithm supports representation learning and node classification for homogeneous graphs. It is an extension of the GCN algorithm that smooths the graph to bring in more distant neighbours of nodes without using multiple layers. |
| (Approximate) Personalized Propagation of Neural Predictions (PPNP/APPNP) [9] | The (A)PPNP algorithm supports fast and scalable representation learning and node classification for attributed homogeneous graphs. In a semi-supervised setting, first a multilayer neural network is trained using the node attributes as input. The predictions from the latter network are then diffused across the graph using a method based on Personalized PageRank. |
| Node2Vec [2] | The Node2Vec and Deepwalk algorithms perform unsupervised representation learning for homogeneous networks, taking into account network structure while ignoring node attributes. The node2vec algorithm is implemented by combining StellarGraph's random walk generator with the word2vec algorithm from [Gensim](https://radimrehurek.com/gensim/). Learned node representations can be used in downstream machine learning models implemented using [Scikit-learn](https://scikit-learn.org/stable/), [Keras](https://keras.io/), [TensorFlow](https://www.tensorflow.org/) or any other Python machine learning library. |
| Metapath2Vec [3] | The metapath2vec algorithm performs unsupervised, metapath-guided representation learning for heterogeneous networks, taking into account network structure while ignoring node attributes. The implementation combines StellarGraph's metapath-guided random walk generator and [Gensim](https://radimrehurek.com/gensim/) word2vec algorithm. As with node2vec, the learned node representations (node embeddings) can be used in downstream machine learning models to solve tasks such as node classification, link prediction, etc, for heterogeneous networks. |
| Relational Graph Convolutional Network [11] | The RGCN algorithm performs semi-supervised learning for node representation and node classification on knowledge graphs. RGCN extends GCN to directed graphs with multiple edge types and works with both sparse and dense adjacency matrices.|
| ComplEx[12] | The ComplEx algorithm computes embeddings for nodes (entities) and edge types (relations) in knowledge graphs, and can use these for link prediction |
| GraphWave [13] | GraphWave calculates unsupervised structural embeddings via wavelet diffusion through the graph. |
| Supervised Graph Classification | A model for supervised graph classification based on GCN [6] layers and mean pooling readout. |
| Watch Your Step [14] | The Watch Your Step algorithm computes node embeddings by using adjacency powers to simulate expected random walks. |
| Deep Graph Infomax [15] | Deep Graph Infomax trains unsupervised GNNs to maximize the shared information between node level and graph level features. |
| Continuous-Time Dynamic Network Embeddings (CTDNE) [16] | Supports time-respecting random walks which can be used in a similar way as in Node2Vec for unsupervised representation learning. |
| DistMult [17] | The DistMult algorithm computes embeddings for nodes (entities) and edge types (relations) in knowledge graphs, and can use these for link prediction |
| DGCNN [18] | The Deep Graph Convolutional Neural Network (DGCNN) algorithm for supervised graph classification. |
| TGCN [19] | The GCN_LSTM model in StellarGraph follows the Temporal Graph Convolutional Network architecture proposed in the TGCN paper with a few enhancements in the layers architecture. |

## Installation

StellarGraph is a Python 3 library and we recommend using Python version `3.6`. The required Python version
can be downloaded and installed from [python.org](https://python.org/). Alternatively, use the Anaconda Python
environment, available from [anaconda.com](https://www.anaconda.com/products/individual#Downloads).

The StellarGraph library can be installed from PyPI, from Anaconda Cloud, or directly from GitHub, as described below.

#### Install StellarGraph using PyPI:
To install StellarGraph library from [PyPI](https://pypi.org) using `pip`, execute the following command:
```
pip install stellargraph
```

[Some of the examples][demos] require installing additional dependencies as well as `stellargraph`. To install these dependencies as well as StellarGraph using `pip` execute the following command:
```
pip install stellargraph[demos]
```

The community detection demos require `python-igraph` which is only available on some platforms. To install this in addition to the other demo requirements:
```
pip install stellargraph[demos,igraph]
```

#### Install StellarGraph in Anaconda Python:
The StellarGraph library is available an [Anaconda Cloud](https://anaconda.org/stellargraph/stellargraph) and can be installed in [Anaconda Python](https://anaconda.com) using the command line `conda` tool, execute the following command:
```
conda install -c stellargraph stellargraph
```


#### Install StellarGraph from GitHub source:
First, clone the StellarGraph repository using `git`:
```
git clone https://github.com/stellargraph/stellargraph.git
```

Then, `cd` to the StellarGraph folder, and install the library by executing the following commands:
```
cd stellargraph
pip install .
```

Some of the examples in the `demos` directory require installing additional dependencies as well as `stellargraph`. To install these dependencies as well as StellarGraph using `pip` execute the following command:
```
pip install .[demos]
```


## Citing
StellarGraph is designed, developed and supported by [CSIRO's Data61](https://data61.csiro.au/).
If you use any part of this library in your research, please cite it using the following BibTex entry
```latex
@misc{StellarGraph,
  author = {CSIRO's Data61},
  title = {StellarGraph Machine Learning Library},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/stellargraph/stellargraph}},
}
```

## References

1. Inductive Representation Learning on Large Graphs. W.L. Hamilton, R. Ying, and J. Leskovec.
Neural Information Processing Systems (NIPS), 2017, ([link](https://arxiv.org/abs/1706.02216) [webpage](https://snap.stanford.edu/graphsage/))

2. Node2Vec: Scalable Feature Learning for Networks. A. Grover, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016, ([link](https://snap.stanford.edu/node2vec/))

3. Metapath2Vec: Scalable Representation Learning for Heterogeneous Networks. Yuxiao Dong, Nitesh V. Chawla, and Ananthram Swami.
ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 135–144, 2017, ([link](https://ericdongyx.github.io/metapath2vec/m2v.html))

4. Attributed Network Embedding via Subspace Discovery. D. Zhang, Y. Jie, X. Zhu and C. Zhang, Data Mining and Knowledge Discovery, 2019, ([link](https://link.springer.com/article/10.1007/s10618-019-00650-2))

5. Graph Attention Networks. P. Veličković et al.
International Conference on Learning Representations (ICLR), 2018, ([link](https://arxiv.org/abs/1710.10903))

6. Graph Convolutional Networks (GCN): Semi-Supervised Classification with Graph Convolutional Networks. Thomas N. Kipf, Max Welling.
International Conference on Learning Representations (ICLR), 2017, ([link](https://github.com/tkipf/gcn))

7. Simplifying Graph Convolutional Networks. F. Wu, T. Zhang, A. H. de Souza, C. Fifty, T. Yu, and K. Q. Weinberger.
International Conference on Machine Learning (ICML), 2019, ([link](https://arxiv.org/abs/1902.07153))

8. Adversarial Examples on Graph Data: Deep Insights into Attack and Defense. H. Wu, C. Wang, Y. Tyshetskiy, A. Docherty, K. Lu, and L. Zhu. IJCAI 2019, ([link](https://arxiv.org/abs/1903.01610))

9. Predict then propagate: Graph neural networks meet personalized PageRank. J. Klicpera, A. Bojchevski, A., and S. Günnemann, ICLR, 2019, arXiv:1810.05997.([link](https://arxiv.org/abs/1810.05997))

10. Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks. W. Chiang, X. Liu, S. Si, Y. Li, S. Bengio, and C. Hsiej, KDD, 2019, arXiv:1905.07953.([link](https://arxiv.org/abs/1905.07953))

11. Modeling relational data with graph convolutional networks. M. Schlichtkrull, T. N. Kipf, P. Bloem, R. Van Den Berg, I. Titov, and M. Welling, European Semantic Web Conference, 2018, arXiv:1609.02907 ([link](https://arxiv.org/pdf/1703.06103.pdf)).

12. Complex Embeddings for Simple Link Prediction. T. Trouillon, J. Welbl, S. Riedel, É. Gaussier and G. Bouchard, ICML, 2016. ([link](http://jmlr.org/proceedings/papers/v48/trouillon16.pdf))

13. Learning Structural Node Embeddings via Diffusion Wavelets. C. Donnat, M. Zitnik, D. Hallac, and J. Leskovec, SIGKDD, 2018, arXiv:1710.10321 ([link](https://arxiv.org/pdf/1710.10321.pdf))

14. Watch Your Step: Learning Node Embeddings via Graph Attention. S. Abu-El-Haija, B. Perozzi, R. Al-Rfou and A. Alemi, NIPS, 2018, arXiv:1710.09599 ([link](https://arxiv.org/pdf/1710.09599.pdf))

15. Deep Graph Infomax. P. Veličković, W. Fedus, W. L. Hamilton, P. Lio, Y. Bengio, R. D. Hjelm. International Conference on Learning Representations (ICLR), 2019, arXiv:1809.10341, ([link](https://arxiv.org/pdf/1809.10341.pdf)).

16. Continuous-Time Dynamic Network Embeddings. Giang Hoang Nguyen, John Boaz Lee, Ryan A. Rossi, Nesreen K. Ahmed, Eunyee Koh, and Sungchul Kim. Proceedings of the 3rd International Workshop on Learning Representations for Big Networks (WWW BigNet) 2018. ([link](https://dl.acm.org/doi/10.1145/3184558.3191526))

17. Embedding Entities and Relations for Learning and Inference in Knowledge Bases. Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng, ICLR, 2015. arXiv:1412.6575 ([link](https://arxiv.org/pdf/1412.6575.pdf))

18. An End-to-End Deep Learning Architecture for Graph Classification. Muhan Zhang, Zhicheng Cui, Marion Neumann, and Yixin Chen, AAAI, 2018. ([link](https://www.cse.wustl.edu/~ychen/public/DGCNN.pdf))

19. T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction. Ling Zhao, Yujiao Song, Chao Zhang, Yu Liu, Pu Wang, Tao Lin, Min Deng, and Haifeng Li. IEEE Transactions on Intelligent Transportation Systems, 2019. ([link](https://ieeexplore.ieee.org/document/8809901))
