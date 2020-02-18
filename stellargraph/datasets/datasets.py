# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Data61, CSIRO
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
`stellargraph.datasets` contains classes to download sample network datasets.

The default download path of ``stellargraph-datasets`` within the user's home directory can be changed by setting the
``STELLARGRAPH_DATASETS_PATH`` environment variable, and each dataset will be downloaded to a subdirectory within this path.
"""

from .dataset_loader import DatasetLoader
from ..core.graph import StellarGraph
import logging
import os
import pandas as pd


log = logging.getLogger(__name__)


class Cora(
    DatasetLoader,
    name="Cora",
    directory_name="cora",
    url="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    url_archive_format="gztar",
    expected_files=["cora.cites", "cora.content"],
    description="The Cora dataset consists of 2708 scientific publications classified into one of seven classes. "
    "The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector "
    "indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words.",
    source="https://linqs.soe.ucsc.edu/data",
):
    pass


class CiteSeer(
    DatasetLoader,
    name="CiteSeer",
    directory_name="citeseer",
    url="https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz",
    url_archive_format="gztar",
    expected_files=["citeseer.cites", "citeseer.content"],
    description="The CiteSeer dataset consists of 3312 scientific publications classified into one of six classes. "
    "The citation network consists of 4732 links. Each publication in the dataset is described by a 0/1-valued word vector "
    "indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 3703 unique words.",
    source="https://linqs.soe.ucsc.edu/data",
):
    pass


class PubMedDiabetes(
    DatasetLoader,
    name="PubMed Diabetes",
    directory_name="Pubmed-Diabetes",
    url="https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz",
    url_archive_format="gztar",
    expected_files=[
        "data/Pubmed-Diabetes.DIRECTED.cites.tab",
        "data/Pubmed-Diabetes.GRAPH.pubmed.tab",
        "data/Pubmed-Diabetes.NODE.paper.tab",
    ],
    description="The PubMed Diabetes dataset consists of 19717 scientific publications from PubMed database "
    "pertaining to diabetes classified into one of three classes. The citation network consists of 44338 links. "
    "Each publication in the dataset is described by a TF/IDF weighted word vector from a dictionary which consists of 500 unique words.",
    source="https://linqs.soe.ucsc.edu/data",
    data_subdirectory_name="data",
):
    pass


class BlogCatalog3(
    DatasetLoader,
    name="BlogCatalog3",
    directory_name="BlogCatalog-dataset",
    url="http://socialcomputing.asu.edu/uploads/1283153973/BlogCatalog-dataset.zip",
    url_archive_format="zip",
    expected_files=[
        "data/edges.csv",
        "data/group-edges.csv",
        "data/groups.csv",
        "data/nodes.csv",
    ],
    description="This dataset is crawled from a social blog directory website BlogCatalog "
    "http://www.blogcatalog.com and contains the friendship network crawled and group memberships.",
    source="http://socialcomputing.asu.edu/datasets/BlogCatalog3",
    data_subdirectory_name="data",
):
    def load(self):
        """
        Load this dataset into an undirected heterogeneous graph, downloading it if required.

        The graph has two types of nodes, 'user' and 'group', and two types of edges, 'friend' and 'belongs'.
        The 'friend' edges connect two 'user' nodes and the 'belongs' edges connects 'user' and 'group' nodes.

        The node and edge types are not included in the dataset that is a collection of node and group ids along with
        the list of edges in the graph.

        Important note about the node IDs: The dataset uses integers for node ids. However, the integers from 1 to 39 are
        used as IDs for both users and groups. This would cause a confusion when constructing the graph object.
        As a result, we convert all IDs to string and append the character 'u' to the integer ID for user nodes and the
        character 'g' to the integer ID for group nodes.

        Returns:
            A :class:`StellarGraph` object.
        """
        self.download()
        return self._load_from_location(self.data_directory)

    @staticmethod
    def _load_from_location(location):
        """
        Support code for the old `load_dataset_BlogCatalog3` function.
        """
        if not os.path.isdir(location):
            raise NotADirectoryError(
                "The location {} is not a directory.".format(location)
            )

        # load the raw data
        user_node_ids = pd.read_csv(os.path.join(location, "nodes.csv"), header=None)
        group_ids = pd.read_csv(os.path.join(location, "groups.csv"), header=None)
        edges = pd.read_csv(
            os.path.join(location, "edges.csv"), header=None, names=["source", "target"]
        )
        group_edges = pd.read_csv(
            os.path.join(location, "group-edges.csv"),
            header=None,
            names=["source", "target"],
        )

        # The dataset uses integers for node ids. However, the integers from 1 to 39 are used as IDs
        # for both users and groups. This is disambiguated by converting everything to strings and
        # prepending u to user IDs, and g to group IDs.
        def u(users):
            return "u" + users.astype(str)

        def g(groups):
            return "g" + groups.astype(str)

        # nodes:
        user_node_ids = u(user_node_ids)
        group_ids = g(group_ids)

        # node IDs in each edge:
        edges = u(edges)
        group_edges["source"] = u(group_edges["source"])
        group_edges["target"] = g(group_edges["target"])

        # arrange the DataFrame indices appropriately: nodes use their node IDs, which have
        # been made distinct above, and the group edges have IDs after the other edges
        user_node_ids.set_index(0, inplace=True)
        group_ids.set_index(0, inplace=True)

        start = len(edges)
        group_edges.index = range(start, start + len(group_edges))

        return StellarGraph(
            nodes={"user": user_node_ids, "group": group_ids},
            edges={"friend": edges, "belongs": group_edges},
        )


class MovieLens(
    DatasetLoader,
    name="MovieLens",
    directory_name="ml-100k",
    url="http://files.grouplens.org/datasets/movielens/ml-100k.zip",
    url_archive_format="zip",
    expected_files=["u.data", "u.user", "u.item", "u.genre", "u.occupation",],
    description="The MovieLens 100K dataset contains 100,000 ratings from 943 users on 1682 movies.",
    source="https://grouplens.org/datasets/movielens/100k/",
):
    pass


class AIFB(
    DatasetLoader,
    name="AIFB",
    directory_name="aifb",
    url="https://ndownloader.figshare.com/files/1118822",
    url_archive_format=None,
    expected_files=["aifbfixed_complete.n3",],
    description="The AIFB dataset describes the AIFB research institute in terms of its staff, research group, and publications. "
    'First used for machine learning with RDF in Bloehdorn, Stephan and Sure, York, "Kernel Methods for Mining Instance Data in Ontologies", '
    "The Semantic Web (2008), http://dx.doi.org/10.1007/978-3-540-76298-0_5. "
    "It contains ~8k entities, ~29k edges, and 45 different relationships or edge types. In (Bloehdorn et al 2007) the dataset "
    "was first used to predict the affiliation (i.e., research group) for people in the dataset. The dataset contains 178 "
    "members of a research group with 5 different research groups. The goal is to predict which research group a researcher belongs to.",
    source="https://figshare.com/articles/AIFB_DataSet/745364",
):
    pass
