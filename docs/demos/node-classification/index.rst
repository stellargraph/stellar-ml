Node classification
======================================

`StellarGraph <https://github.com/stellargraph/stellargraph>`_ provides numerous algorithms for doing node classification on graphs. This folder contains demos of all of them to explain how they work and how to use them as part of a TensorFlow Keras data science workflow.

A node classification task predicts an attribute of each node in a graph. For instance, labelling each node with a categorical class (binary classification or multiclass classification), or predicting a continuous number (regression). It is supervised or semi-supervised, where the model is trained using a subset of nodes that have ground-truth labels.

Node classification can also be done as a downstream task from node representation learning/embeddings, by training a supervised or semi-supervised classifier against the embedding vectors. Unsupervised algorithms that can be used in this manner include random walk-based methods like Metapath2Vec. StellarGraph provides :doc:`demos of unsupervised algorithms <../embeddings/index>`, some of which include a node classification downstream task.

Find algorithms and demos for a graph
-------------------------------------

This table lists all node classification demos, including the algorithms trained, the types of graph used, and the tasks demonstrated.

.. list-table::
   :header-rows: 1

   * - Demo
     - Algorithm(s)
     - Node features
     - Heterogeneous
     - Directed
     - Edge weights
     - Inductive
     - Node embeddings
   * - :doc:`GCN <gcn-node-classification>`
     - GCN
     - yes
     -
     -
     -
     -
     - yes
   * - :doc:`Cluster-GCN <cluster-gcn-node-classification>`
     - Cluster-GCN
     - yes
     -
     -
     -
     -
     - yes
   * - :doc:`RGCN <rgcn-node-classification>`
     - RGCN
     - yes
     - yes, multiple edge types
     -
     -
     -
     - yes
   * - :doc:`GAT <gat-node-classification>`
     - GAT
     - yes
     -
     -
     -
     -
     - yes
   * - :doc:`SGC <sgc-node-classification>`
     - SGC
     - yes
     -
     -
     -
     -
     - yes
   * - :doc:`PPNP & APPNP <ppnp-node-classification>`
     - PPNP, APPNP
     - yes
     -
     -
     -
     -
     -
   * - :doc:`Attri2Vec <attri2vec-node-classification>`
     - Attri2Vec
     - yes
     -
     -
     -
     -
     - yes
   * - :doc:`GraphSAGE on Cora <graphsage-node-classification>`
     - GraphSAGE
     - yes
     -
     -
     -
     -
     - yes
   * - :doc:`Inductive GraphSAGE <graphsage-inductive-node-classification>`
     - GraphSAGE
     - yes
     -
     -
     -
     - yes
     - yes
   * - :doc:`Directed GraphSAGE <directed-graphsage-node-classification>`
     - GraphSAGE
     - yes
     -
     - yes
     -
     -
     - yes
   * - :doc:`Node2Vec <node2vec-node-classification>`
     - Node2Vec
     -
     -
     -
     -
     -
     - yes
   * - :doc:`Weighted Node2Vec <node2vec-weighted-node-classification>`
     - Node2Vec
     -
     -
     -
     - yes
     -
     - yes


See :doc:`the root README <../../README>` or each algorithm's documentation for the relevant citation(s). See :doc:`the demo index <../index>` for more tasks, and a summary of each algorithm.

Table of contents
-----------------

.. toctree::
    :titlesonly:
    :glob:

    ./*
