import pandas as pd
import holoviews as hv
from pathlib import Path

from .extraction_instrument import ExtractionInstrument


class Reporter:

    def __init__(self, data_file_path: Path, figures_path: Path) -> None:
        self.__extraction_instrument = ExtractionInstrument(
            data_file_path=data_file_path
        )
        self.__figures_path = figures_path
        hv.extension("bokeh")

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
        heatmap = hv.HeatMap(class_numbers).opts(
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
        input_data = self.__extraction_instrument.input_data.drop(
            columns="Demographics Questions"
        )
        features = self.__extraction_instrument.features
        feature_selection = self.__extraction_instrument.models[
            ["Use Feature Selection"]
        ]
        models = self.__extraction_instrument.models.drop(
            columns=["Use Feature Selection", "Ensemble of Models"]
        )
        model_ensembling = self.__extraction_instrument.models[["Ensemble of Models"]]
        print(~input_data.isna())
        print(~features.isna())
        print(~feature_selection.isna())
        print(~models.isna())
        print(~model_ensembling.isna())
        exit()
        print(df)
        chart = hv.Sankey(data=df)
        hv.save(chart, f"{self.__figures_path}/classification_pipeline.html")

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
