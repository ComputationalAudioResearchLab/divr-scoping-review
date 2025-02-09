import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import holoviews as hv
import circlify as circ
import matplotlib.colorbar
from pathlib import Path
import matplotlib.pyplot as plt
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

    def pipeline_per_label(self):
        class_numbers = self.__extraction_instrument.class_usage(min_usage=5)
        print(class_numbers)
        C1 = self.__extraction_instrument.co_occurence(
            X=class_numbers, Y=self.__extraction_instrument.input_data
        )
        print(C1)
        C2 = self.__extraction_instrument.co_occurence(
            X=class_numbers, Y=self.__extraction_instrument.features
        )
        print(C2)
        C3 = self.__extraction_instrument.co_occurence(
            X=class_numbers, Y=self.__extraction_instrument.models
        )
        print(C3)

        np.set_printoptions(linewidth=100, precision=2)
        # pdf = C1.pivot(index="Y", columns="X", values="frequency")
        # A = pdf.to_numpy()
        # print(A)
        # B = np.tile(A.max(axis=0)[:, None], (1, A.shape[1]))
        # print(B)
        # normalized_A = A / B
        # print(normalized_A)
        # exit()

        def heatmap(df: pd.DataFrame, ax, cmap, y_label: str):
            df = df.pivot(index="Y", columns="X", values="frequency").fillna(0)
            sns.heatmap(data=df, ax=ax, annot=True, cmap=cmap)
            ax.set_xlabel(None)
            ax.set_ylabel(y_label, labelpad=20, fontsize=18)
            ax.tick_params(labelsize=14)

        fig, ax = plt.subplots(
            3,
            1,
            figsize=(10, 15),
            constrained_layout=True,
            sharex=True,
        )
        heatmap(df=C1, ax=ax[0], cmap="YlGn", y_label="Input Data Type")
        heatmap(df=C2, ax=ax[1], cmap="GnBu", y_label="Feature")
        heatmap(df=C3, ax=ax[2], cmap="BuPu", y_label="Model")
        ax[2].set_xlabel("Diagnostic Label", fontsize=18)
        fig.suptitle(
            "Frequency of usage of data type/feature/model per diagnostic label",
            fontsize=20,
            y=1.03,
        )
        lines = [(1.0060, "#00000040"), (0.727, "#00000020"), (0.455, "#00000020")]
        for y, c in lines:
            fig.add_artist(
                plt.Line2D(
                    [0.00, 1.0],
                    [y] * 2,
                    color=c,
                    lw=2,
                    transform=fig.transFigure,
                )
            )
        fig.canvas.draw()
        self.align_labels(axes_list=ax, axis="y")
        fig.savefig(
            f"{self.__figures_path}/pipeline_per_label.png",
            bbox_inches="tight",
        )

    def align_labels(self, axes_list, axis="y", align=None):
        if align is None:
            align = "l" if axis == "y" else "b"
        yx, xy = [], []
        for ax in axes_list:
            yx.append(ax.yaxis.label.get_position()[0])
            xy.append(ax.xaxis.label.get_position()[1])

        if axis == "x":
            if align in ("t", "top"):
                lim = max(xy)
            elif align in ("b", "bottom"):
                lim = min(xy)
        else:
            if align in ("l", "left"):
                lim = min(yx)
            elif align in ("r", "right"):
                lim = max(yx)

        if align in ("t", "b", "top", "bottom"):
            for ax in axes_list:
                t = ax.xaxis.label.get_transform()
                x, y = ax.xaxis.label.get_position()
                ax.xaxis.set_label_coords(x, lim, t)
        else:
            for ax in axes_list:
                t = ax.yaxis.label.get_transform()
                x, y = ax.yaxis.label.get_position()
                ax.yaxis.set_label_coords(lim, y, t)

    def classification_circles_2(self):
        pipeline = self.__extraction_instrument.classification_pipeline
        pipeline = pipeline.dropna(subset=["Input Data"])
        print(pipeline)

        group = ["Input Data", "Input Feature", "Model"]
        counts = pipeline.groupby(by=group)["Model"].count()
        df = pd.merge(
            left=pipeline.drop_duplicates(subset=group),
            right=counts,
            left_on=group,
            right_index=True,
        )
        df = df[group + ["Model_y"]].rename(columns={"Model_y": "frequency"})
        print(df)
        df["Input Data"], input_data_encoding = self.label_encode(df["Input Data"])
        # df["Input Feature"], input_feature_encoding = self.label_encode(
        #     df["Input Feature"]
        # )
        df["Model"], model_encoding = self.label_encode(df["Model"])
        df = df[["Input Data", "Input Feature", "Model", "frequency"]]
        x_tick_labels = list(input_data_encoding.keys())
        y_tick_labels = list(model_encoding.keys())
        print(df)

        def packed_circles(row: pd.DataFrame):
            row = row.sort_values(by="frequency", ascending=False, inplace=False)
            radii = [1] * len(row)
            circs = circ.circlify(data=radii)
            scale_factor = radii[0] / circs[0].r
            circles = []
            for circle, label, freq in zip(
                circs, row["Input Feature"], row["frequency"]
            ):
                x = circle.x
                y = circle.y
                r = circle.r
                circles += [{"x": x, "y": y, "r": r, "label": label, "frequency": freq}]
            return pd.Series({"square_side": scale_factor * 2, "circles": circles})

        groups = df.groupby(by=["Input Data", "Model"])
        print("max packing in square: ", groups["Input Feature"].apply(len).max())
        res = (
            groups[["Input Feature", "frequency"]]
            .apply(packed_circles)
            .explode("circles")
        )
        print(res)
        res = (
            res.reset_index()
            .join(pd.json_normalize(res["circles"]))
            .drop(columns="circles")
        )
        res["radius_scale_factor"] = res["square_side"] / res["square_side"].max()
        res["x"] = (res["x"] * res["radius_scale_factor"]) + (res["Input Data"] * 2)
        res["y"] = (res["y"] * res["radius_scale_factor"]) + (res["Model"] * 2)
        res["r"] = res["r"] * res["radius_scale_factor"]
        label_keys = {
            "MFCC": "a",
            "Mel Spectrogram": "b",
            "Other audio processing features": "c",
            "Label Encoding": "d",
            "EGG Spectrogram": "e",
            "Wavelets": "f",
            "Glottal signal": "g",
            "NN features": "h",
            "Raw Audio": "i",
            "Bespoke Algorithm": "j",
        }
        res["label_key"] = res["label"].apply(label_keys.get)
        print(res)
        print(res["radius_scale_factor"].max())

        cmap = plt.colormaps["YlGnBu"]
        cmap_r = plt.colormaps["YlGnBu_r"]
        # hue_encoding = self.label_encoding(ser=res["label"])
        # palette = sns.palettes.color_palette(
        #     palette="rocket", n_colors=len(hue_encoding)
        # )
        width = res["x"].max() - res["x"].min()
        height = res["y"].max() - res["y"].min()
        max_frequency = res["frequency"].max()
        print(max_frequency)
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        print(res["square_side"].max())
        for row_idx, row in res.iterrows():
            x = row["x"]
            y = row["y"]
            r = row["r"]
            label = row["label"]
            label_key = row["label_key"]
            frequency = row["frequency"]
            # color = palette[hue_encoding[label]]
            ratio = frequency / max_frequency
            circle = plt.Circle((x, y), r, color=cmap(ratio), alpha=0.99, label=label)
            ax.add_patch(circle)
            fontshift = 0.1
            fontcolor = "black" if ratio < 0.5 else "white"
            plt.text(
                x=x - fontshift,
                y=y - fontshift,
                s=label_key,
                color=fontcolor,
                fontdict={"size": 18},
            )
        for x in range(0, 23, 2):
            for y in range(0, 27, 2):
                circle = plt.Circle((x, y), 1, color="black", alpha=0.025)
                ax.add_patch(circle)
        padding = 1
        ax.set_xlim(0 - padding, 22 + padding)
        ax.set_ylim(0 - padding, 26 + padding)
        ax.set_xticks(
            ticks=range(0, int(width + padding * 2), 2),
            labels=x_tick_labels,
            rotation=90,
        )
        ax.set_yticks(
            ticks=range(0, int(height + padding * 2), 2),
            labels=y_tick_labels,
            rotation=0,
        )
        ax.grid(
            # color='green',
            # linestyle="--",
            linewidth=1,
            alpha=0.5,
        )
        # Fake grid
        # for y in range(-1, 27, 1):
        #     if (y % 2) == 0:
        #         alpha = 0.25
        #         plt.axhline(y=y, color="black", alpha=alpha, linewidth=1)
        # for x in range(-1, 23, 1):
        #     if (y % 2) == 0:
        #         alpha = 0.25
        #         plt.axvline(x=x, color="black", alpha=alpha, linewidth=1)
        ax.tick_params(labelsize=24)
        # c_ax = plt.axes([0.915, 0.11, 0.01, 0.77])
        c_ax = plt.axes([0.995, 0.11, 0.01, 0.77])
        matplotlib.colorbar.Colorbar(
            ax=c_ax,
            cmap=cmap,
            values=range(1, max_frequency + 1),
            ticks=range(1, max_frequency + 1),
        )
        c_ax.tick_params(labelsize=24)

        hist_x_axis = plt.axes(
            [0.125, 0.885, 0.775, 0.077],
        )
        hist_x_axis.hist(
            x=df["Input Data"],
            bins=12,
            orientation="vertical",
            color="#A1E3F9",
        )
        hist_x_axis.set_ylim(0, 60)
        ticks = [0, 30, 60]
        hist_x_axis.set_yticks(ticks=ticks, labels=ticks)
        ticks = np.linspace(0.47, 10.53, 12)
        hist_x_axis.set_xticks(
            ticks=ticks,
            labels=ticks,
            rotation=90,
            alpha=0.0,
        )
        hist_x_axis.tick_params(labelsize=24)
        hist_x_axis.margins(0)
        hist_x_axis.grid(
            # color='green',
            # linestyle="--",
            linewidth=1,
            alpha=0.5,
        )

        hist_y_axis = plt.axes([0.905, 0.11, 0.085, 0.77])
        hist_y_axis.hist(
            x=df["Model"],
            bins=14,
            orientation="horizontal",
            color="#CFEE91",
        )

        ticks = [0, 15, 30]
        hist_y_axis.set_xlim(0, 30)
        hist_y_axis.set_xticks(ticks=ticks, labels=ticks, rotation=90)
        hist_y_axis.margins(0)
        ticks = np.linspace(0.47, 12.53, 14)
        hist_y_axis.set_yticks(
            ticks=ticks,
            labels=ticks,
            rotation=90,
            alpha=0.0,
        )
        hist_y_axis.tick_params(labelsize=24)
        hist_y_axis.margins(0)
        hist_y_axis.grid(
            # color='green',
            # linestyle="--",
            linewidth=1,
            alpha=0.5,
        )

        fig.savefig(f"{self.__figures_path}/circles_2.png", bbox_inches="tight")
        # print(df.to_numpy())

    def classification_circles(self):
        pipeline = self.__extraction_instrument.classification_pipeline
        pipeline = pipeline.dropna(subset=["Input Data"])
        print(pipeline)

        group = ["Input Data", "Input Feature", "Model"]
        counts = pipeline.groupby(by=group)["Model"].count()
        df = pd.merge(
            left=pipeline.drop_duplicates(subset=group),
            right=counts,
            left_on=group,
            right_index=True,
        )
        df = df[group + ["Model_y"]].rename(columns={"Model_y": "frequency"})
        print(df)
        df["Input Data"], input_data_encoding = self.label_encode(df["Input Data"])
        # df["Input Feature"], input_feature_encoding = self.label_encode(
        #     df["Input Feature"]
        # )
        df["Model"], model_encoding = self.label_encode(df["Model"])
        df = df[["Input Data", "Input Feature", "Model", "frequency"]]
        print(df)

        def packed_circles(row: pd.DataFrame):
            row = row.sort_values(by="frequency", ascending=False, inplace=False)
            radii = row["frequency"].tolist()
            circs = circ.circlify(data=radii)
            scale_factor = radii[0] / circs[0].r
            circles = []
            for circle, label in zip(circs, row["Input Feature"]):
                x = circle.x
                y = circle.y
                r = circle.r
                circles += [{"x": x, "y": y, "r": r, "label": label}]
            return pd.Series({"square_side": scale_factor * 2, "circles": circles})

        groups = df.groupby(by=["Input Data", "Model"])
        print("max packing in square: ", groups["Input Feature"].apply(len).max())
        res = (
            groups[["Input Feature", "frequency"]]
            .apply(packed_circles)
            .explode("circles")
        )
        print(res)
        res = (
            res.reset_index()
            .join(pd.json_normalize(res["circles"]))
            .drop(columns="circles")
        )
        res["radius_scale_factor"] = res["square_side"] / res["square_side"].max()
        res["x"] = (res["x"] * res["radius_scale_factor"]) + (res["Input Data"] * 2)
        res["y"] = (res["y"] * res["radius_scale_factor"]) + (res["Model"] * 2)
        res["r"] = res["r"] * res["radius_scale_factor"]
        print(res)
        print(res["radius_scale_factor"].max())

        # cmap = plt.colormaps["coolwarm_r"]
        hue_encoding = self.label_encoding(ser=res["label"])
        palette = sns.palettes.color_palette(
            palette="rocket", n_colors=len(hue_encoding)
        )
        width = res["x"].max() - res["x"].min()
        height = res["y"].max() - res["y"].min()
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        print(res["square_side"].max())
        min_radius = res["r"].max() * 0.005  # 0.5% of biggest radius
        for row_idx, row in res.iterrows():
            x = row["x"]
            y = row["y"]
            r = row["r"]
            label = row["label"]
            color = palette[hue_encoding[label]]
            # color = cmap(row["r"])
            if r >= min_radius:
                circle = plt.Circle((x, y), r, color=color, alpha=0.99, label=label)
                ax.add_patch(circle)
            else:
                plt.text(x=x, y=y, s="x", color=color)
        padding = 1
        ax.set_xlim(res["x"].min() - padding, res["x"].max() + padding)
        ax.set_ylim(res["y"].min() - padding, res["y"].max() + padding)
        ax.set_xticks(ticks=range(int(width + padding * 2)))
        ax.set_yticks(ticks=range(int(height + padding * 2)))
        # ax.margins(0)
        ax.grid(
            # color='green',
            # linestyle="--",
            linewidth=0.5,
        )
        fig.savefig(f"{self.__figures_path}/circles.png", bbox_inches="tight")
        # print(df.to_numpy())

    def label_encode(
        self, ser: "pd.Series[str]"
    ) -> tuple["pd.Series[int]", dict[int, str]]:
        encoding = self.label_encoding(ser=ser)
        encoded_ser = ser.apply(encoding.get)
        return (encoded_ser, encoding)

    def label_encoding(self, ser: "pd.Series[str]") -> dict[int, str]:
        return {v: i for i, v in enumerate(sorted(ser.unique().tolist()))}

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
                    edges += [(article_idx, model, model_ensembling, edge_value, None)]
        edges = pd.DataFrame.from_records(
            data=edges,
            columns=["article_idx", "source", "target", "edge_value", "attribute"],
        )
        G = self.__network_graph(nodes=nodes, edges=edges)
        self.__hv_nx(G=G)

        # self.__sankey_hv_png(nodes=nodes, edges=edges)
        # self.__sankey_plotly(nodes=nodes, edges=edges)

    def __horizontal_pipeline_layout(self, G: nx.Graph) -> dict[int, list[float]]:
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
                    "Class Weights",
                    "Explicitly Balanced",
                    "Minor Imbalance",
                    "Unspecified",
                    "Major Imbalance",
                ],
            },
            "Input Feature": {
                "x": 0.40,
                "y": [
                    "MFCC",
                    "Label Encoding",
                    "Mel Spectrogram",
                    "NN features",
                    "Glottal signal",
                    "Bespoke Algorithm",
                    "Raw Audio",
                    "Wavelets",
                    "EGG Spectrogram",
                    "Other audio processing features",
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
                    "Decision Trees",
                    "Linear/Logistic Regression",
                    "RF",
                    "HMM",
                    "ANN (simple NN arch)",
                    "Naïve Bayes",
                    "Gradient Boosting",
                    "GMM",
                    "PCA/LDA/DA",
                    "kNN/Clustering",
                    "Voting Classifier",
                    "Bayes Net",
                    "DNN (complex NN arch)",
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

    def __free_pipeline_layout(self, G: nx.Graph) -> dict[int, list[float]]:
        xy_map = {
            "Input Data": {
                "Vowel /a/": [0 / 5, 11 / 11],
                "Vowel /e/": [0 / 5, 10 / 11],
                "Vowel /i/": [0 / 5, 9 / 11],
                "Vowel /u/": [0 / 5, 8 / 11],
                "/pataka/": [0 / 5, 7 / 11],
                "Rainbow Passage": [0 / 5, 6 / 11],
                "Sentence(s)": [0 / 5, 5 / 11],
                "Multi-sentence passage": [0 / 5, 4 / 11],
                "Repeated Word": [0 / 5, 3 / 11],
                "Demographic questions": [0 / 5, 2 / 11],
                "EGG": [0 / 5, 1 / 11],
                "Unspecified": [0 / 5, 0 / 11],
                # "Vowel /a/": [0 / 5, 11 / 11],
                # "Vowel /e/": [-0.1 / 5, 10 / 11],
                # "Vowel /i/": [-0.2 / 5, 9 / 11],
                # "Vowel /u/": [-0.3 / 5, 8 / 11],
                # "/pataka/": [-0.4 / 5, 7 / 11],
                # "Rainbow Passage": [-0.5 / 5, 6 / 11],
                # "Sentence(s)": [-0.50 / 5, 5 / 11],
                # "Multi-sentence passage": [-0.4 / 5, 4 / 11],
                # "Repeated Word": [-0.3 / 5, 3 / 11],
                # "Demographic questions": [-0.2 / 5, 2 / 11],
                # "EGG": [-0.1 / 5, 1 / 11],
                # "Unspecified": [0 / 5, 0 / 11],
            },
            "Data Balancing": {
                # "Already Balanced": [0.8 / 5, 8 / 10],
                # "Class Weights": [0.7 / 5, 7 / 10],
                # "Explicitly Balanced": [0.6 / 5, 6 / 10],
                # "Minor Imbalance": [0.6 / 5, 5 / 10],
                # "Unspecified": [0.7 / 5, 4 / 10],
                # "Major Imbalance": [0.8 / 5, 3 / 10],
                "Already Balanced": [-0.1, 0],
                "Class Weights": [-0.1, 0],
                "Explicitly Balanced": [-0.1, 0],
                "Minor Imbalance": [-0.1, 0],
                "Unspecified": [-0.1, 0],
                "Major Imbalance": [-0.1, 0],
            },
            "Input Feature": {
                "MFCC": [2 / 5, 9 / 9],
                "Label Encoding": [2 / 5, 8 / 9],
                "Mel Spectrogram": [2 / 5, 7 / 9],
                "NN features": [2 / 5, 6 / 9],
                "Glottal signal": [2 / 5, 5 / 9],
                "Bespoke Algorithm": [2 / 5, 4 / 9],
                "Raw Audio": [2 / 5, 3 / 9],
                "Wavelets": [2 / 5, 2 / 9],
                "EGG Spectrogram": [2 / 5, 1 / 9],
                "Other audio processing features": [2 / 5, 0 / 9],
            },
            "Feature Selection": {
                # "Feature Selection": [3 / 5, 1 / 1],
                "Feature Selection": [-0.1, 0],
            },
            "Model": {
                "SVM": [4 / 5, 13 / 13],
                "Decision Trees": [4 / 5, 12 / 13],
                "Linear/Logistic Regression": [4 / 5, 11 / 13],
                "RF": [4 / 5, 10 / 13],
                "HMM": [4 / 5, 9 / 13],
                "ANN (simple NN arch)": [4 / 5, 8 / 13],
                "Naïve Bayes": [4 / 5, 7 / 13],
                "Gradient Boosting": [4 / 5, 6 / 13],
                "GMM": [4 / 5, 5 / 13],
                "PCA/LDA/DA": [4 / 5, 4 / 13],
                "kNN/Clustering": [4 / 5, 3 / 13],
                "Voting Classifier": [4 / 5, 2 / 13],
                "Bayes Net": [4 / 5, 1 / 13],
                "DNN (complex NN arch)": [4 / 5, 0 / 13],
            },
            "Model Ensembling": {
                "Model Ensembling": [5 / 5, 1 / 1],
            },
        }
        layout = {}
        for node_key, node_data in G.nodes.items():
            node_type = node_data["type"]
            node_label = node_data["label"]
            layout[node_key] = xy_map[node_type][node_label]
        return layout

    def __hv_nx(self, G: nx.Graph) -> None:
        hv.extension("bokeh")
        pos = self.__horizontal_pipeline_layout(G)
        # pos = self.__free_pipeline_layout(G)
        pos_xy = np.array(list(pos.values()))
        pos_xy[:, 1] += 0.025
        pos_labels = [node["label"] for node in G.nodes.values()]
        edge_cmap = sns.color_palette("crest", n_colors=len(G.edges) + 1).as_hex()
        chart = hv.Graph.from_networkx(G, pos).opts(
            edge_color="article_idx",
            edge_cmap=edge_cmap,
            edge_alpha=1.00,
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
