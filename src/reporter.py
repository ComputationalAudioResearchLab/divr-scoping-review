import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import holoviews as hv
from pathlib import Path
from bokeh.io import export_svgs
import plotly.graph_objects as go

from .extraction_instrument import ExtractionInstrument


class Reporter:

    def __init__(self, data_file_path: Path, figures_path: Path) -> None:
        self.__extraction_instrument = ExtractionInstrument(
            data_file_path=data_file_path
        )
        self.__figures_path = figures_path

    def classification_labels(self):
        # minimum articles that must use a given class
        min_usage_of_class = 5
        class_numbers = self.__extraction_instrument.diagnostic_class_numbers
        class_numbers["counts"] = self.__extraction_instrument.diagnostic_class_counts
        article_indices_ticks = [(i, i + 1) for i in range(0, len(class_numbers))]
        class_numbers = class_numbers.drop(columns=["counts"])
        column_names = class_numbers.columns.tolist()

        class_presence = class_numbers.notna().astype(int)
        dists = sorted(
            [(class_presence[c].sum(), c) for c in column_names], key=lambda x: x[0]
        )
        selected_columns = [x[1] for x in dists if x[0] > min_usage_of_class]
        class_numbers = class_numbers[selected_columns]

        class_numbers = class_numbers.reset_index().melt(
            id_vars="index", var_name="Diagnostic Label", value_name="Count"
        )
        class_numbers = class_numbers.rename(columns={"index": "Article Index"})
        hv.extension("bokeh")
        heatmap = hv.HeatMap(class_numbers).opts(
            colorbar=True,
            width=1600,
            height=550,
            xticks=article_indices_ticks,
            xrotation=90,
            show_grid=True,
            title="Sample count of diagnostic labels per article",
            fontsize={
                "title": 25,
                "labels": 20,
                "xticks": 12,
                "yticks": 20,
                "cticks": 15,
            },
            gridstyle={
                "grid_line_color": "black",
                "grid_line_width": 20,
            },
        )
        hv.save(
            obj=heatmap, filename=f"{self.__figures_path}/classification_labels.html"
        )

    def classification_pipeline(self):
        pipeline = self.__extraction_instrument.classification_pipeline
        main_file_index = pipeline["Main File Index"].notna().to_numpy()
        start_indices = main_file_index.nonzero()[0]
        end_indices = np.concatenate((start_indices[1:], [len(main_file_index)]))
        pipeline["Feature Selection"] = pipeline["Feature Selection"].astype(bool)
        pipeline["Model Ensembling"] = pipeline["Model Ensembling"].astype(bool)

        nodes = []
        node_labels = {
            "Input Data": set(pipeline["Input Data"].dropna().tolist()),
            "Data Balancing": set(pipeline["Data Balancing"].dropna().tolist()),
            "Input Feature": set(pipeline["Input Feature"].dropna().tolist()),
            "Feature Selection": ["Feature Selection"],
            "Model": set(pipeline["Model"].dropna().tolist()),
            "Model Ensembling": ["Model Ensembling"],
        }
        lookup = {}
        node_idx = 0
        for type_idx, (node_type, labels) in enumerate(node_labels.items()):
            if node_type not in lookup:
                lookup[node_type] = {}
            for subtype_idx, label in enumerate(labels):
                nodes += [(node_type, label, node_idx, type_idx + 1, subtype_idx + 1)]
                lookup[node_type][label] = node_idx
                node_idx += 1

        nodes = pd.DataFrame.from_records(
            data=nodes, columns=["type", "label", "node_idx", "type_idx", "subtype_idx"]
        )
        nodes["type_idx"] = nodes["type_idx"] / (nodes["type_idx"].max() + 2)
        nodes["subtype_idx"] = nodes["subtype_idx"] / (nodes["subtype_idx"].max() + 2)

        for key in ["Input Data", "Data Balancing", "Input Feature", "Model"]:
            pipeline[key] = pipeline[key].apply(
                lambda x: lookup[key][x] if pd.notna(x) else pd.NA
            )
        for key in ["Feature Selection", "Model Ensembling"]:
            pipeline[key] = pipeline[key].apply(
                lambda x: lookup[key][key] if x else pd.NA
            )

        edges = []
        for idx, (start, end) in enumerate(zip(start_indices, end_indices)):
            article_idx = idx + 1
            # line at [start] position is article name + main file index, hence omitted
            article = pipeline.iloc[start + 1 : end]
            article = article[
                [
                    "Input Data",
                    "Data Balancing",
                    "Input Feature",
                    "Feature Selection",
                    "Model",
                    "Model Ensembling",
                ]
            ]
            article = article.drop_duplicates(
                subset=["Input Data", "Input Feature", "Model"]
            )
            edge_value = 1 / len(article)
            for row_idx, row in article.iterrows():
                input_data = row["Input Data"]
                data_balancing = row["Data Balancing"]
                input_feature = row["Input Feature"]
                feature_selection = row["Feature Selection"]
                model_ensembling = row["Model Ensembling"]
                model = row["Model"]
                edges += [
                    (article_idx, input_data, data_balancing, edge_value),
                    (article_idx, data_balancing, input_feature, edge_value),
                ]
                if pd.notna(feature_selection):
                    edges += [
                        (article_idx, input_feature, feature_selection, edge_value),
                        (article_idx, feature_selection, model, edge_value),
                    ]
                else:
                    edges += [(article_idx, input_feature, model, edge_value)]

                if pd.notna(model_ensembling):
                    edges += [(article_idx, model, model_ensembling, edge_value)]
        edges = pd.DataFrame.from_records(
            data=edges, columns=["article_idx", "source", "target", "edge_value"]
        )
        G = self.__network_graph(nodes=nodes, edges=edges)
        self.__hv_nx(G=G)

        # self.__sankey_hv_png(nodes=nodes, edges=edges)
        # self.__sankey_plotly(nodes=nodes, edges=edges)

    def __custom_pipeline_layout(self, G: nx.Graph) -> dict[int, list[float]]:
        xy_map = {
            "Input Data": {
                "x": 0.00,
                "y": [
                    "Vowel /a/",
                    "Vowel /e/",
                    "Vowel /i/",
                    "Vowel /u/",
                    "/pataka/",
                    "Rainbow Passage",
                    "Sentence(s)",
                    "Multi-sentence passage",
                    "Repeated Word",
                    "Demographic questions",
                    "EGG",
                    "Unspecified",
                ],
            },
            "Data Balancing": {
                "x": 0.20,
                "y": [
                    "Already Balanced",
                    "Explicitly Balanced",
                    "Major Imbalance",
                    "Class Weights",
                    "Minor Imbalance",
                    "Unspecified",
                ],
            },
            "Input Feature": {
                "x": 0.40,
                "y": [
                    "MFCC",
                    "Glottal signal",
                    "Other audio processing features",
                    "Label Encoding",
                    "Mel Spectrogram",
                    "NN features",
                    "Bespoke Algorithm",
                    "Raw Audio",
                    "Wavelets",
                    "EGG Spectrogram",
                ],
            },
            "Feature Selection": {
                "x": 0.60,
                "y": [
                    "Feature Selection",
                ],
            },
            "Model": {
                "x": 0.80,
                "y": [
                    "SVM",
                    "DNN (complex NN arch)",
                    "ANN (simple NN arch)",
                    "Decision Trees",
                    "Linear/Logistic Regression",
                    "RF",
                    "HMM",
                    "Naïve Bayes",
                    "Gradient Boosting",
                    "GMM",
                    "PCA/LDA/DA",
                    "kNN/Clustering",
                    "Voting Classifier",
                    "Bayes Net",
                ],
            },
            "Model Ensembling": {
                "x": 1.00,
                "y": [
                    "Model Ensembling",
                ],
            },
        }
        layout = {}
        for node_key, node_data in G.nodes.items():
            node_type = node_data["type"]
            node_label = node_data["label"]
            xy = xy_map[node_type]
            x = xy["x"]
            max_label_idx = len(xy["y"]) - 1
            if max_label_idx > 0:
                label_idx = xy["y"].index(node_label)
                y = (max_label_idx - label_idx) / max_label_idx
            else:
                y = 1.00
            layout[node_key] = [x, y]
        return layout

    def __hv_nx(self, G: nx.Graph) -> None:
        hv.extension("bokeh")
        pos = self.__custom_pipeline_layout(G)
        pos_xy = np.array(list(pos.values()))
        pos_xy[:, 1] += 0.025
        pos_labels = [node["label"] for node in G.nodes.values()]
        edge_cmap = sns.color_palette("crest", n_colors=len(G.edges) + 1).as_hex()
        chart = hv.Graph.from_networkx(G, pos).opts(
            edge_color="article_idx",
            edge_cmap=edge_cmap,
        )
        labels = hv.Labels(
            {("x", "y"): pos_xy, "labels": pos_labels},
            ["x", "y"],
            "labels",
        ).opts(text_font_size="12pt", text_color="black")
        chart = chart * labels
        chart = chart.opts(
            # colorbar=True,
            width=1600,
            height=900,
            # xticks=article_indices_ticks,
            # xrotation=90,
            show_grid=True,
            # title="Sample count of diagnostic labels per article",
            # fontsize={
            #     "title": 25,
            #     "labels": 20,
            #     "xticks": 12,
            #     "yticks": 20,
            #     "cticks": 15,
            # },
            # gridstyle={
            #     "grid_line_color": "black",
            #     "grid_line_width": 20,
            # },
        )
        hv.save(chart, f"{self.__figures_path}/classification_pipeline.html")
        plot_state = hv.renderer("bokeh").get_plot(chart).state
        plot_state.output_backend = "svg"
        export_svgs(
            plot_state, filename=f"{self.__figures_path}/classification_pipeline.svg"
        )

    def __network_graph(self, nodes: pd.DataFrame, edges: pd.DataFrame):
        G = nx.Graph()
        nodes.apply(
            lambda row: G.add_node(
                row["node_idx"],
                type=row["type"],
                label=row["label"],
            ),
            axis=1,
        )
        edges.apply(
            lambda row: G.add_edge(
                row["source"],
                row["target"],
                article_idx=row["article_idx"],
            ),
            axis=1,
        )
        return G

    def __sankey_plotly(self, nodes: pd.DataFrame, edges: pd.DataFrame):
        # [0.0, 0.17, 0.33, 0.5, 0.67, 0.83]
        x_map = {
            "Input Data": 0.00,
            "Data Balancing": 0.20,
            "Input Feature": 0.40,
            "Feature Selection": 0.60,
            "Model": 0.80,
            "Model Ensembling": 1.0,
        }
        link_palette = sns.color_palette(
            "rocket", n_colors=max(edges["article_idx"]) + 1
        )
        link_rgb = [[int(c * 255) for c in colors] for colors in link_palette]
        node_palette = sns.color_palette("crest", n_colors=6).as_hex()
        color_map = {
            "Input Data": node_palette[0],
            "Data Balancing": node_palette[1],
            "Input Feature": node_palette[2],
            "Feature Selection": node_palette[3],
            "Model": node_palette[4],
            "Model Ensembling": node_palette[5],
        }
        X = nodes["type"].apply(x_map.get).tolist()
        Y = nodes["type"].apply(x_map.get).tolist()
        colors = nodes["type"].apply(color_map.get).tolist()
        X = [0.001 if v == 0 else 0.999 if v == 1 else v for v in X]
        Y = [0.001 if v == 0 else 0.999 if v == 1 else v for v in Y]
        print(X)
        print(Y)

        link_colors = [
            f"rgba({link_rgb[idx][0]},{link_rgb[idx][1]},{link_rgb[idx][2]},{0.2})"
            for idx, ev in zip(edges["article_idx"], edges["edge_value"])
        ]
        print(link_colors[:10])

        node = {
            "label": nodes["label"].tolist(),
            "x": X,
            # "y": Y,
            # "color": colors,
        }
        link = {
            "source": edges["source"],
            "target": edges["target"],
            "value": edges["edge_value"],
            # "color": link_colors,
            # "color": [f"rgba(0,0,0,{e})" for e in edges["edge_value"]],
        }
        fig = go.Figure(go.Sankey(arrangement="freeform", node=node, link=link))
        fig.data[0].node.x = X
        # fig.data[0].node.y = Y
        fig.write_image(f"{self.__figures_path}/classification_pipeline.png")
        fig.write_html(f"{self.__figures_path}/classification_pipeline.html")

    def __sankey_hv_png(self, nodes, edges):
        hv.extension("matplotlib")
        chart = hv.Sankey(edges).opts(
            colorbar=True,
            width=1600,
            height=900,
            title="Classification Pipeline",
            # fontsize={
            #     "title": 25,
            #     "labels": 20,
            #     "xticks": 12,
            #     "yticks": 20,
            #     "cticks": 15,
            # },
            # gridstyle={
            #     "grid_line_color": "black",
            #     "grid_line_width": 20,
            # },
        )
        hv.save(chart, f"{self.__figures_path}/classification_pipeline.png")

    def print_extraction_instrument(self):
        print(self.__extraction_instrument.train_test_set_sizes)
        print(self.__extraction_instrument.models)
        print(self.__extraction_instrument.features)
        print(self.__extraction_instrument.input_data)
        print(self.__extraction_instrument.diagnostic_class_numbers)
        print(self.__extraction_instrument.demographics)
        print(self.__extraction_instrument.groups_of_diagnostic_labels)
        print(self.__extraction_instrument.database_usage)
        print(self.__extraction_instrument.database_characteristics)
        print(self.__extraction_instrument.protocol_checklist)
