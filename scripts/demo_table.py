# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
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

import argparse
import contextlib
import os.path
import subprocess
import sys

SEPARATOR = "\n<!-- DEMO TABLE MARKER -->\n"
HTML_INDENT = 2
LINK_DEFAULT_TEXT = "demo"
TRUE_TEXT = "yes"


class HtmlBuilder:
    def __init__(self, indent=None):
        self.html = []
        self.indent_amount = indent
        self.indent_level = 0
        self.add_count = 0

    def add(self, data, one_line=False):
        self.add_count += 1
        if one_line:
            self.html[-1] += data
        else:
            if self.indent_amount:
                indent = " " * (self.indent_amount * self.indent_level)
                data = indent + data

            self.html.append(data)

    @contextlib.contextmanager
    def element(self, name, attrs={}, only_with_attrs=False):
        """Open (and automatically) close an HTML element"""
        if only_with_attrs and not attrs:
            yield
            return

        attrs_str = " ".join(f"{name}='{value}'" for name, value in attrs.items())
        if attrs_str:
            attrs_str = " " + attrs_str

        self.add(f"<{name}{attrs_str}>")
        self.indent_level += 1
        initial_len = len(self.html)
        try:
            yield
        finally:
            self.indent_level -= 1
            closing = f"</{name}>"
            self.add(f"</{name}>", one_line=len(self.html) == initial_len)

    def string(self):
        sep = "" if self.indent_amount is None else "\n"
        return sep.join(self.html)


class T:
    def __init__(self, text=None, link=None, details=None):
        if text is None:
            if link is None:
                raise ValueError("must specify at least one of 'text' and 'link'")

            text = LINK_DEFAULT_TEXT

        self.text = text
        self.link = link
        self.details = details

    @staticmethod
    def textify(inp):
        if not inp:
            return None
        elif inp is True:
            return T(TRUE_TEXT)
        elif isinstance(inp, list):
            return [T.textify(x) for x in inp]
        elif isinstance(inp, T):
            return inp

        return T(inp)

    def link_is_valid_relative(self, base_dir):
        if self.link is None:
            return True

        if os.path.isabs(self.link):
            # absolute links aren't allowed
            return False

        if self.link.lower().startswith("http"):
            # github (and other website) links aren't allowed either
            return False

        return os.path.exists(os.path.join(base_dir, self.link))

    def to_html(self):
        html = HtmlBuilder()

        title_attr = {"title": self.details} if self.details else {}
        href_attr = {"href": self.link} if self.link else {}

        # add a span if we need details (hover) text, and an link if there's a link
        with html.element("span", title_attr, only_with_attrs=True):
            with html.element("a", href_attr, only_with_attrs=True):
                html.add(self.text)

        return html.string()

    def to_rst(self):
        if self.link:
            return f":any:`{self.text} <{self.link}>`"

        return self.text


def validate_links(element, base_dir, invalid):
    # traverse over the collection(s) to find the T's and check their links, collecting them in
    # `invalid`.
    if element is None:
        pass
    elif isinstance(element, T):
        if not element.link_is_valid_relative(base_dir):
            invalid.append(element.link)
    elif isinstance(element, list):
        for sub in element:
            validate_links(sub, base_dir, invalid)
    elif isinstance(element, Algorithm):
        for sub in element.columns.values():
            validate_links(sub, base_dir, invalid)
    else:
        raise ValueError(f"unsupported element in link validation {element!r}")


def cell_html(html, cell):
    if not cell:
        return

    if isinstance(cell, list):
        for contents in cell:
            # multiple elements? space them out
            html.add(contents.to_html(), one_line=False)
    else:
        html.add(cell.to_html(), one_line=True)

def cell_rst(cell):
    if not cell:
        return ""

    if isinstance(cell, list):
        return " ".join(contents.to_rst() for contents in cell)
    else:
        return cell.to_rst()


def build_html(headings, algorithms):
    builder = HtmlBuilder(indent=2)

    builder.add(
        f"<!-- autogenerated by {__file__}, edit that file instead of this location -->"
    )
    with builder.element("table"):
        with builder.element("tr"):
            for heading in headings:
                with builder.element("th"):
                    builder.add(heading.to_html(), one_line=True)

        for algorithm in algorithms:
            with builder.element("tr"):
                for heading in headings:
                    with builder.element("td"):
                        cell_html(builder, algorithm.columns[heading])

    return builder.string()

def build_rst(headings, algorithms):
    result = [".. list-table::", "   :header-rows: 1", ""]

    new_row = "   *"
    new_item = "     - "

    result.append(new_row)
    for heading in headings:
        result.append(new_item + heading.to_rst())

    for algorithm in algorithms:
        result.append(new_row)
        for heading in headings:
            result.append(new_item + cell_rst(algorithm.columns[heading]))

    return "\n".join(result)


# Columns
ALGORITHM = T("Algorithm")
HETEROGENEOUS = T("Heter.", details="Heterogeneous graphs")
DIRECTED = T("Dir.", details="Directed")
WEIGHTED = T("EW", details="Edge weights")
TEMPORAL = T("T", details="Time-varying or temporal graphs")
FEATURES = T("NF", details="Node features")
NC = T("NC", link="node-classification/README.md", details="Node classification")
LP = T("LP", link="link-prediction/README.md", details="Link prediction")
RL = T("RL", link="embeddings/README.md", details="Representation learning")
INDUCTIVE = T("Ind.", details="Inductive")
GC = T("GC", link="graph-classification/README.md", details="Graph classification")

COLUMNS = [
    ALGORITHM,
    HETEROGENEOUS,
    DIRECTED,
    WEIGHTED,
    TEMPORAL,
    FEATURES,
    NC,
    LP,
    RL,
    INDUCTIVE,
    GC,
]


class Algorithm:
    def __init__(
        self,
        algorithm,
        *,
        heterogeneous=None,
        directed=None,
        weighted=None,
        temporal=None,
        features=None,
        nc=None,
        interpretability_nc=None,
        lp=None,
        rl=None,
        inductive=None,
        gc=None,
    ):
        columns = {
            ALGORITHM: algorithm,
            HETEROGENEOUS: heterogeneous,
            DIRECTED: directed,
            WEIGHTED: weighted,
            TEMPORAL: temporal,
            FEATURES: features,
            NC: nc,
            LP: lp,
            RL: rl,
            INDUCTIVE: inductive,
            GC: gc,
        }

        self.columns = {name: T.textify(value) for name, value in columns.items()}


HETEROGENEOUS_EDGE = T("yes, edges", details="Multiple edges types and one node type")


def rl_us(link=None):
    return T("US", link=link, details="UnsupervisedSampler, using link prediction")


def rl_dgi(link="embeddings/deep-graph-infomax-cora.ipynb"):
    return T("DGI", link=link, details="DeepGraphInfomax, using mutual information")


def via_rl(link=None):
    return T(
        "via RL",
        link=link,
        details="As a downstream task by training a classifier on reprentation/embedding vectors",
    )


ALGORITHMS = [
    Algorithm(
        T("GCN", details="Graph Convolutional Network"),
        heterogeneous="see RGCN",
        features=True,
        nc=T(link="node-classification/gcn/gcn-cora-node-classification-example.ipynb"),
        interpretability_nc=T(
            link="interpretability/gcn/node-link-importance-demo-gcn.ipynb"
        ),
        lp=T(link="link-prediction/gcn/cora-gcn-links-example.ipynb"),
        rl=[rl_us(), rl_dgi()],
        inductive="see Cluster-GCN",
        gc=T(link="graph-classification/supervised-graph-classification.ipynb"),
    ),
    Algorithm(
        "Cluster-GCN",
        features=True,
        nc=T(
            link="node-classification/cluster-gcn/cluster-gcn-node-classification.ipynb"
        ),
        lp=True,
        inductive=True,
    ),
    Algorithm(
        T("RGCN", details="Relational GCN"),
        heterogeneous=HETEROGENEOUS_EDGE,
        features=True,
        nc=T(
            link="node-classification/rgcn/rgcn-aifb-node-classification-example.ipynb"
        ),
        lp=True,
    ),
    Algorithm(
        T("GAT", details="Graph ATtention Network"),
        features=True,
        nc=T(link="node-classification/gat/gat-cora-node-classification-example.ipynb"),
        interpretability_nc=T(
            link="interpretability/gat/node-link-importance-demo-gat.ipynb"
        ),
        lp=True,
        rl=[rl_us(), rl_dgi()],
    ),
    Algorithm(
        T("SGC", details="Simplified Graph Convolution"),
        features=True,
        nc=T(link="node-classification/sgc/sgc-node-classification-example.ipynb"),
        lp=True,
    ),
    Algorithm(
        T("PPNP", details="Personalized Propagation of Neural Predictions"),
        features=True,
        nc=T(
            link="node-classification/ppnp/ppnp-cora-node-classification-example.ipynb"
        ),
        lp=True,
        rl=[rl_us(), rl_dgi(link=None)],
    ),
    Algorithm(
        T("APPNP", details="Approximate PPNP"),
        features=True,
        nc=T(
            link="node-classification/ppnp/ppnp-cora-node-classification-example.ipynb"
        ),
        lp=True,
        rl=[rl_us(), rl_dgi()],
    ),
    Algorithm(
        "GraphWave",
        nc=via_rl(),
        lp=via_rl(),
        rl=T(link="embeddings/graphwave-barbell.ipynb"),
    ),
    Algorithm(
        "Attri2Vec",
        features=True,
        nc=T(
            link="node-classification/attri2vec/attri2vec-citeseer-node-classification-example.ipynb"
        ),
        lp=T(link="link-prediction/attri2vec/stellargraph-attri2vec-DBLP.ipynb"),
        rl=T(link="embeddings/stellargraph-attri2vec-citeseer.ipynb"),
    ),
    Algorithm(
        "GraphSAGE",
        heterogeneous="see HinSAGE",
        directed=T(
            link="node-classification/graphsage/directed-graphsage-on-cora-example.ipynb"
        ),
        features=True,
        nc=T(
            link="node-classification/graphsage/graphsage-cora-node-classification-example.ipynb"
        ),
        lp=T(link="link-prediction/graphsage/cora-links-example.ipynb"),
        rl=[
            rl_us(link="embeddings/embeddings-unsupervised-graphsage-cora.ipynb"),
            rl_dgi(),
        ],
        inductive=T(
            link="node-classification/graphsage/graphsage-pubmed-inductive-node-classification-example.ipynb"
        ),
    ),
    Algorithm(
        "HinSAGE",
        heterogeneous=True,
        features=True,
        nc=True,
        lp=T(link="link-prediction/hinsage/movielens-recommender.ipynb"),
        rl=rl_dgi(),
        inductive=True,
    ),
    Algorithm(
        "Node2Vec",
        weighted=T(
            link="node-classification/node2vec/stellargraph-node2vec-weighted-random-walks.ipynb"
        ),
        nc=via_rl(
            link="node-classification/node2vec/stellargraph-node2vec-node-classification.ipynb"
        ),
        lp=via_rl(link="link-prediction/random-walks/cora-lp-demo.ipynb"),
        rl=T(link="embeddings/stellargraph-node2vec.ipynb"),
    ),
    Algorithm(
        "Metapath2Vec",
        heterogeneous=True,
        nc=via_rl(),
        lp=via_rl(),
        rl=T(link="embeddings/stellargraph-metapath2vec.ipynb"),
    ),
    Algorithm(
        T("CTDNE", details="Continuous-Time Dynamic Network Embeddings"),
        temporal=True,
        nc=via_rl(),
        lp=via_rl(link="link-prediction/random-walks/ctdne-link-prediction.ipynb"),
        rl=True,
    ),
    Algorithm(
        "Watch Your Step",
        nc=via_rl(link="embeddings/watch-your-step-cora-demo.ipynb"),
        lp=via_rl(),
        rl=T(link="embeddings/watch-your-step-cora-demo.ipynb"),
    ),
    Algorithm(
        "ComplEx",
        heterogeneous=HETEROGENEOUS_EDGE,
        directed=True,
        nc=via_rl(),
        lp=T(link="link-prediction/knowledge-graphs/complex.ipynb"),
        rl=True,
    ),
    Algorithm(
        "DistMult",
        heterogeneous=HETEROGENEOUS_EDGE,
        directed=True,
        nc=via_rl(),
        lp=T(link="link-prediction/knowledge-graphs/distmult.ipynb"),
        rl=True,
    ),
]


def main():
    parser = argparse.ArgumentParser(
        description="Edits or compares the table of all algorithms and their demos in `demos/README.md`."
    )
    parser.add_argument(
        "readme",
        type=argparse.FileType("r+"),
        default="demos/README.md",
        nargs="?",
        help="the location of the readme file (default: %(default)s)",
    )
    parser.add_argument(
        "--action",
        choices=["compare", "overwrite"],
        default="compare",
        help="whether to compare the table in 'readme' against what would be generated, or to overwrite the table with a new one (default: %(default)s)",
    )
    args = parser.parse_args()

    def error(message, edit_fixit=False):
        formatted = f"Error while generating algorithm table for `{args.readme.name}`: {message}"
        if edit_fixit:
            formatted += f"\n\nTo fix, edit `{__file__}` as appropriate and run it like `python {__file__} --action=overwrite` to overwrite existing table with new one."

        print(formatted, file=sys.stderr)

        try:
            subprocess.run(
                [
                    "buildkite-agent",
                    "annotate",
                    "--style=error",
                    "--context=demo_table",
                    formatted,
                ]
            )
        except FileNotFoundError:
            # no agent, so probably on buildkite, and so silently no annotation
            pass

        sys.exit(1)

    base_dir = os.path.dirname(args.readme.name)

    invalid_links = []
    validate_links(COLUMNS, base_dir, invalid_links)
    validate_links(ALGORITHMS, base_dir, invalid_links)

    if invalid_links:
        formatted = "\n".join(f"- `{link}`" for link in invalid_links)
        error(
            f"expected all links in algorithm specifications in `{__file__}` to be relative links that are valid starting at `{base_dir}`, but found {len(invalid_links)} invalid:\n\n{formatted}",
            edit_fixit=True,
        )

    rst_table = build_rst(COLUMNS, ALGORITHMS)
    print(rst_table)
    return

    new_table = build_html(COLUMNS, ALGORITHMS)

    file_contents = args.readme.read()
    parts = file_contents.split(SEPARATOR)
    if len(parts) != 3:
        error(
            f"expected exactly two instances of `{SEPARATOR.strip()}` on their own lines in `{args.readme.name}`, found {len(parts) - 1} instances."
        )

    prefix, current_table, suffix = parts

    if args.action == "compare":
        if new_table != current_table:
            error(
                f"existing table in `{args.readme.name}` differs to generated table; was it edited manually?",
                edit_fixit=True,
            )

    elif args.action == "overwrite":
        file_chunks = [prefix, SEPARATOR, new_table, SEPARATOR, suffix]

        args.readme.seek(0)
        args.readme.write("".join(file_chunks))

        # delete any remaining content
        args.readme.truncate()


if __name__ == "__main__":
    main()
