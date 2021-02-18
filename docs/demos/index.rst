StellarGraph demos
==================

`StellarGraph <https://github.com/stellargraph/stellargraph>`_ provides numerous algorithms for graph machine learning. This folder contains demos of all of them to explain how they work and how to use them as part of a TensorFlow Keras data science workflow.

The demo notebooks can be run without any installation of Python by using Binder or Google Colab - these both provide a cloud-based notebook environment.  The whole set of demos can be opened in Binder |binder| or you can click the Binder and Colab badges within each notebook.

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/stellargraph/stellargraph/master?urlpath=lab/tree/demos/README.md
   :alt: Open in Binder


Find algorithms for a task
--------------------------


* Introduction to StellarGraph and its graph machine learning workflow (with TensorFlow and Keras): :doc:`GCN on Cora <node-classification/gcn-node-classification>`
* Predicting attributes, such as classifying as a class or label, or regressing to calculate a continuous number:

  * For nodes/vertices/entities: :doc:`node classification <./node-classification/index>`
  * For edges/links/connections: :doc:`link prediction <./link-prediction/index>` (includes knowledge graph completion)
  * For graphs/networks: :doc:`graph classification <./graph-classification/index>`
  * Adjusting predictions scores to be probabilities (for any model): :doc:`calibration <./calibration/index>`
  * Interpreting/introspecting models, for node classification: :doc:`interpretability <./interpretability/index>`

* Representation learning or computing embedding vectors (including unsupervised tasks):

  * For nodes/vertices/entities, edges/links/connections and graphs: :doc:`embeddings <./embeddings/index>`

* Time series or sequence prediction for nodes within a graph (including spatio-temporal data): :doc:`time series <./time-series/index>`
* Ensembling models to reduce prediction variance: :doc:`ensembles <./ensembles/index>`
* Loading data into a ``StellarGraph`` object, with Pandas, NumPy, Neo4j or NetworkX: :doc:`basics <./basics/index>`
* Experimental: running GraphSAGE or Cluster-GCN on data stored in Neo4j: :doc:`neo4j connector <./connector/neo4j/index>`

Find a demo for an algorithm
----------------------------

..
   DEMO TABLE MARKER
.. list-table::
   :header-rows: 1

   *
     - Algorithm
     - Heterogeneous
     - Directed
     - Edge weights
     - Time-varying, temporal
     - Node features
     - :any:`Node classification <node-classification/index>`
     - :any:`Link prediction <link-prediction/index>`
     - :any:`Unsupervised <embeddings/index>`
     - Inductive
     - :any:`Graph classification <graph-classification/index>`
   *
     - Graph Convolutional Network (GCN)
     - see RGCN
     -
     - yes
     - see T-GCN
     - yes
     - :any:`demo <node-classification/gcn-node-classification>`
     - :any:`demo <link-prediction/gcn-link-prediction>`
     - UnsupervisedSampler, :any:`DeepGraphInfomax <embeddings/deep-graph-infomax-embeddings>`
     - via Cluster-GCN
     - :any:`demo <graph-classification/gcn-supervised-graph-classification>`
   *
     - Cluster-GCN
     -
     -
     - yes
     -
     - yes
     - :any:`demo <node-classification/cluster-gcn-node-classification>`
     - yes
     - :any:`DeepGraphInfomax <embeddings/deep-graph-infomax-embeddings>`
     - yes
     -
   *
     - Relational GCN (RGCN)
     - multiple edges types
     -
     - yes
     -
     - yes
     - :any:`demo <node-classification/rgcn-node-classification>`
     - yes
     - :any:`DeepGraphInfomax <embeddings/deep-graph-infomax-embeddings>`
     -
     -
   *
     - Temporal GCN (T-GCN), implemented as GCN-LSTM
     -
     -
     -
     - node features
     - time series, sequence
     - :any:`demo <time-series/gcn-lstm-time-series>`
     -
     -
     -
     -
   *
     - Graph ATtention Network (GAT)
     -
     -
     - yes
     -
     - yes
     - :any:`demo <node-classification/gat-node-classification>`
     - yes
     - UnsupervisedSampler, :any:`DeepGraphInfomax <embeddings/deep-graph-infomax-embeddings>`
     - via Cluster-GCN
     -
   *
     - Simplified Graph Convolution (SGC)
     -
     -
     - yes
     -
     - yes
     - :any:`demo <node-classification/sgc-node-classification>`
     - yes
     -
     -
     -
   *
     - Personalized Propagation of Neural Predictions (PPNP)
     -
     -
     - yes
     -
     - yes
     - :any:`demo <node-classification/ppnp-node-classification>`
     - yes
     - UnsupervisedSampler, DeepGraphInfomax
     -
     -
   *
     - Approximate PPNP (APPNP)
     -
     -
     - yes
     -
     - yes
     - :any:`demo <node-classification/ppnp-node-classification>`
     - yes
     - UnsupervisedSampler, :any:`DeepGraphInfomax <embeddings/deep-graph-infomax-embeddings>`
     - via Cluster-GCN
     -
   *
     - GraphWave
     -
     -
     - yes
     -
     -
     - via embedding vectors
     - via embedding vectors
     - :any:`demo <embeddings/graphwave-embeddings>`
     -
     -
   *
     - Attri2Vec
     -
     -
     -
     -
     - yes
     - :any:`demo <node-classification/attri2vec-node-classification>`
     - :any:`demo <link-prediction/attri2vec-link-prediction>`
     - :any:`demo <embeddings/attri2vec-embeddings>`
     - yes
     -
   *
     - GraphSAGE
     - see HinSAGE
     - :any:`demo <node-classification/directed-graphsage-node-classification>`
     - yes
     -
     - yes
     - :any:`demo <node-classification/graphsage-node-classification>`
     - :any:`demo <link-prediction/graphsage-link-prediction>`
     - :any:`UnsupervisedSampler <embeddings/graphsage-unsupervised-sampler-embeddings>`, :any:`DeepGraphInfomax <embeddings/deep-graph-infomax-embeddings>`
     - :any:`demo <node-classification/graphsage-inductive-node-classification>`
     -
   *
     - HinSAGE
     - yes
     -
     -
     -
     - yes
     - yes
     - :any:`demo <link-prediction/hinsage-link-prediction>`
     - :any:`DeepGraphInfomax <embeddings/deep-graph-infomax-embeddings>`
     - yes
     -
   *
     - Node2Vec
     -
     -
     - :any:`demo <node-classification/node2vec-weighted-node-classification>`
     -
     -
     - via embedding vectors, :any:`keras layer <node-classification/keras-node2vec-node-classification>`, :any:`gensim <node-classification/node2vec-node-classification>`
     - :any:`via embedding vectors <link-prediction/node2vec-link-prediction>`
     - :any:`keras layer <embeddings/keras-node2vec-embeddings>`, :any:`gensim <embeddings/node2vec-embeddings>`
     -
     -
   *
     - Metapath2Vec
     - yes
     -
     -
     -
     -
     - via embedding vectors
     - :any:`via embedding vectors <link-prediction/metapath2vec-link-prediction>`
     - :any:`demo <embeddings/metapath2vec-embeddings>`
     -
     -
   *
     - Continuous-Time Dynamic Network Embeddings
     -
     -
     -
     - yes
     -
     - via embedding vectors
     - :any:`via embedding vectors <link-prediction/ctdne-link-prediction>`
     - yes
     -
     -
   *
     - Watch Your Step
     -
     -
     - yes
     -
     -
     - :any:`via embedding vectors <embeddings/watch-your-step-embeddings>`
     - via embedding vectors
     - :any:`demo <embeddings/watch-your-step-embeddings>`
     -
     -
   *
     - ComplEx
     - multiple edges types
     - yes
     -
     -
     -
     - via embedding vectors
     - :any:`demo <link-prediction/complex-link-prediction>`
     - yes
     -
     -
   *
     - DistMult
     - multiple edges types
     - yes
     -
     -
     -
     - via embedding vectors
     - :any:`demo <link-prediction/distmult-link-prediction>`
     - yes
     -
     -
   *
     - Deep Graph CNN
     -
     -
     - yes
     -
     - yes
     -
     -
     -
     -
     - :any:`demo <graph-classification/dgcnn-graph-classification>`
..
   DEMO TABLE MARKER

See :doc:`the root README <../README>` or each algorithm's documentation for the relevant citation(s).


Download the demos
------------------

You can run download a local copy of the demos using the :code:`curl` command below:

.. code-block:: console

    curl -L https://github.com/stellargraph/stellargraph/archive/master.zip | tar -xz --strip=1 stellargraph-master/demos

The dependencies required to run most of our demo notebooks locally can be installed using one of the following:

- Using ``pip``: :code:`pip install stellargraph[demos]`
- Using ``conda``: :code:`conda install -c stellargraph stellargraph`

Table of contents
-----------------

.. toctree::
    :titlesonly:
    :glob:

    */index
