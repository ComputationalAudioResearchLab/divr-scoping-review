import pandas as pd


from .extraction_instrument import ExtractionInstrument


class Reporter:

    def __init__(self) -> None:
        self.__extraction_instrument = ExtractionInstrument()

    def classification_pipeline(self):
        df = pd.concat(
            [
                self.__extraction_instrument.input_data.drop(
                    columns="Demographics Questions"
                ),
                self.__extraction_instrument.features,
                self.__extraction_instrument.models.drop(
                    columns=["Use Feature Selection", "Ensemble of Models"]
                ),
            ]
        )
        print(df)

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
