import numpy as np
import pandas as pd
import holoviews as hv
from pathlib import Path
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
            "Input Data": set(pipeline["Input Data"].tolist()),
            "Data Balancing": set(pipeline["Data Balancing"].tolist()),
            "Input Feature": set(pipeline["Input Feature"].tolist()),
            "Feature Selection": ["Feature Selection"],
            "Model": set(pipeline["Model"].tolist()),
            "Model Ensembling": ["Model Ensembling"],
        }
        lookup = {}
        node_idx = 0
        for type_idx, (node_type, labels) in enumerate(node_labels.items()):
            if node_type not in lookup:
                lookup[node_type] = {}
            for subtype_idx, label in enumerate(labels):
                nodes += [(node_type, label, node_idx, type_idx, subtype_idx)]
                lookup[node_type][label] = node_idx
                node_idx += 1

        nodes = pd.DataFrame.from_records(
            data=nodes, columns=["type", "label", "node_idx", "type_idx", "subtype_idx"]
        )
        nodes["type_idx"] = nodes["type_idx"] / nodes["type_idx"].max()
        nodes["subtype_idx"] = nodes["subtype_idx"] / nodes["subtype_idx"].max()

        for key in ["Input Data", "Data Balancing", "Input Feature", "Model"]:
            pipeline[key] = pipeline[key].apply(lambda x: lookup[key][x])
        for key in ["Feature Selection", "Model Ensembling"]:
            pipeline[key] = pipeline[key].apply(
                lambda x: lookup[key][key] if x else pd.NA
            )

        edges = []
        for idx, (start, end) in enumerate(zip(start_indices, end_indices)):
            # article_idx = idx + 1
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
            edge_value = 1 / len(article)
            for row_idx, row in article.iterrows():
                input_data = row["Input Data"]
                data_balancing = row["Data Balancing"]
                input_feature = row["Input Feature"]
                feature_selection = row["Feature Selection"]
                model_ensembling = row["Model Ensembling"]
                model = row["Model"]
                edges += [
                    (input_data, data_balancing, edge_value),
                    (data_balancing, input_feature, edge_value),
                ]
                if pd.notna(feature_selection):
                    edges += [
                        (input_feature, feature_selection, edge_value),
                        (feature_selection, model, edge_value),
                    ]
                else:
                    edges += [(input_feature, model, edge_value)]

                if pd.notna(model_ensembling):
                    edges += [(model, model_ensembling, edge_value)]
        edges = pd.DataFrame.from_records(
            data=edges, columns=["source", "target", "edge_value"]
        )
        # self.__sankey_hv_png(nodes=nodes, edges=edges)
        self.__sankey_plotly(nodes=nodes, edges=edges)

    def __sankey_plotly(self, nodes: pd.DataFrame, edges: pd.DataFrame):
        # [0.0, 0.17, 0.33, 0.5, 0.67, 0.83]
        x_map = {
            "Input Data": 0.00,
            "Data Balancing": 0.20,
            "Input Feature": 0.30,
            "Feature Selection": 0.50,
            "Model": 0.70,
            "Model Ensembling": 0.8,
        }
        node = {
            "label": nodes["label"].tolist(),
            "x": nodes["type"].apply(x_map.get).tolist(),
            # "y": nodes["type"].apply(x_map.get).tolist(),
            # "x": nodes["type_idx"],
            "y": nodes["subtype_idx"],
        }
        link = {
            "source": edges["source"],
            "target": edges["target"],
            "value": edges["edge_value"],
        }
        fig = go.Figure()
        fig.add_sankey(arrangement="freeform", node=node, link=link)
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
