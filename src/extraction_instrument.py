import pandas as pd


from pathlib import Path


class ExtractionInstrument:

    __sheets = [
        "Main",
        "Classification Pipeline",
        "Train test set sizes",
        "Models",
        "Features",
        "Input Data",
        "Diagnostic class numbers",
        "Demographics",
        "Groups of diagnostic labels",
        "Database Usage",
        "Database Characteristics",
        "Protocol Checklist",
    ]

    __num_articles = 74

    def __init__(self, data_file_path: Path) -> None:
        self.__workbook = pd.read_excel(data_file_path, sheet_name=self.__sheets)

    @property
    def main(self):
        return self.__workbook["Main"]

    @property
    def best_accuracy(self):
        return self.__workbook["Main"]["Best multi-class classification accuracy"][
            : self.__num_articles
        ]

    @property
    def classification_pipeline(self):
        return self.__workbook["Classification Pipeline"]

    @property
    def data_balancing(self):
        pipeline = self.__workbook["Classification Pipeline"][
            ["Main File Index", "Data Balancing"]
        ]
        main_file_index = pipeline["Main File Index"].notna().to_numpy()
        start_indices = main_file_index.nonzero()[0]
        return pipeline["Data Balancing"].iloc[start_indices + 1].reset_index(drop=True)

    @property
    def train_test_set_sizes(self):
        return self.__workbook["Train test set sizes"].iloc[: self.__num_articles][
            [
                "Training/Testing Splits",
                "Total Data",
                "Training (%)",
                "Testing (%)",
                "Validation (%)",
                "Tuning (%)",
                "Training",
                "Testing",
                "Validation",
                "Tuning",
            ]
        ]

    @property
    def models(self):
        return self.__workbook["Models"].iloc[: self.__num_articles][
            [
                # "Use Feature Selection",
                "Ensemble of Models",
                "DNN (complex NN arch)",
                "ANN (simple NN arch)",
                "SVM",
                "HMM",
                "GMM",
                "kNN/Clustering",
                "RF",
                "Decision Trees",
                "Gradient Boosting",
                "Linear/Logistic Regression",
                "Naïve Bayes",
                "BayesNet",
                "Voting Classifier",
                "PCA/LDA/DA",
            ]
        ]

    @property
    def features(self):
        return (
            self.__workbook["Features"]
            .iloc[: self.__num_articles][
                [
                    "Raw Audio",
                    "NN Features",
                    "Mel Spectrogram",
                    "MFCC",
                    "Wavelets",
                    "Glottal signal",
                    # "Other audio processing features",
                    "Pitch and Fundamental Frequency (F0)",
                    "Frequency-Related Features (Spectral Characteristics)",
                    "Voice Quality (Phonation, Noise & Regularity)",
                    "Intensity & Loudness",
                    "Temporal & Rhythm Features",
                    "Cepstral Features",
                    "Irregular Voicing ",
                    "Prosody & Fluency",
                    "Demographics/questionnaire",
                    "Custom",
                ]
            ]
            .rename(
                columns={
                    "Pitch and Fundamental Frequency (F0)": "Pitch",
                    "Frequency-Related Features (Spectral Characteristics)": "Spectral Characteristics",
                    "Voice Quality (Phonation, Noise & Regularity)": "Voice Quality",
                    "Temporal & Rhythm Features": "Temporal & Rhythm",
                    "Irregular Voicing ": "Irregular Voicing",
                    "Prosody & Fluency": "Prosody & Fluency",
                }
            )
        )

    @property
    def input_data(self):
        return self.__workbook["Input Data"].iloc[: self.__num_articles][
            [
                "Vowel /a/",
                "Vowel /e/",
                "Vowel /i/",
                "Vowel /u/",
                "/pataka/",
                "Rainbow Passage",
                "Sentence(s)",
                "Multi-sentence passage",
                "Repeated Word",
                "Demographics",
                "EGG",
                "Unspecified",
            ]
        ]

    @property
    def diagnostic_class_counts(self):
        return self.__workbook["Diagnostic class numbers"].iloc[: self.__num_articles][
            "Count of different classes"
        ]

    @property
    def count_per_diagnostic_label(self):
        return (
            self.diagnostic_class_numbers.stack()
            .reset_index()
            .groupby("level_1")[0]
            .sum()
            .reset_index()
            .rename(columns={"level_1": "label", 0: "count"})
        )

    @property
    def diagnostic_class_numbers(self):
        return self.__workbook["Diagnostic class numbers"].iloc[: self.__num_articles][
            [
                "A-P Squeezing",
                "Adductor spasmodic Dysphonia",
                "Apraxia of Speech",
                "Benign Mucosal Disease",
                "Central laryngeal motion disorder",
                "Cleft Lip Palatte",
                "Contact Pachyderma",
                "Cordectomy",
                "Cysts",
                "Depression",
                "Dysarthria",
                "Dysphonia",
                "Euphony",
                "Friedreich Ataxia",
                "Front Lateral Partial Resection",
                "Functional Dysphonia",
                "Gastric Reflux",
                "Glottic Neoplasm",
                "Hyperfunction",
                "Hyperfunctional Dysphonia",
                "Hyperkinetic Dysphonia",
                "Hyperthyroidism",
                "Hypofunction",
                "Hypokinetic Dysphonia",
                "Hypothyroidism",
                "Keratosis",
                "Laryngeal cancer",
                "Laryngitis Chronica",
                "Laryngeal Neoplasm",
                "Laryngeal Pathologies",
                "Laryngitis",
                "Leukoplakia",
                "Mass Lesions",
                "Morphological Alteration",
                "Multiple System Atrophy",
                "Multiple Sclerosis",
                "Muscle Tension",
                "Neoplasm",
                "Neurological Disorder",
                "Neuromuscular",
                "Nodules",
                "Non-Structural Dysphonia",
                "Normal",
                "Oedemas",
                "Organic",
                "Organic & Functional",
                "Organofunctional",
                "Organic Vocal Fold Lesions",
                "Other vocal diseases",
                "Others (Nodules, Cysts, Vocal Fold Paralysis)",
                "Parkinsons",
                "Pathological disorders",
                "Phonotrauma",
                "Physiological disorders",
                "Polyps",
                "Progressive supranuclear palsy",
                "Recurrent paresis",
                "Recurrent Laryngeal Nerve Paralysis",
                "Reflux Laryngitis",
                "Renkei's Edema",
                "Spasmodic Dysphonia",
                "Structural",
                "Sulcus Vocalis",
                "Unilateral Vocal Paralysis",
                "Ventricular Compression",
                "Vocal Atrophy",
                "Vocal fold edemas or nodules",
                "Vocal Fold Paralysis",
                "Vocal fold paresis",
                "Vocal Nodules, Polyps, Cysts",
                "Vocal Palsy",
                "Vox Senilis",
            ]
        ]

    def diagnositic_class_numbers_comorbities(self):
        return self.__workbook["Diagnostic class numbers"].iloc[: self.__num_articles][
            ["Dysphonia + Laryngitis", "Laryngitis + Reinke's Edema"]
        ]

    @property
    def demographics(self):
        return self.__workbook["Demographics"].iloc[: self.__num_articles][
            ["Demographics of participants", "Male (count)", "Female (count)"]
        ]

    @property
    def groups_of_diagnostic_labels(self):
        return self.__workbook["Groups of diagnostic labels"].iloc[
            : self.__num_articles
        ][
            [
                "Constructs supersets (from base classes in data)",
                "Benign mucosal disease",
                "Benign Phonotraumatic Lesions (Phonotrauma)",
                "Functional",
                "Inflammatory",
                "Laryngeal Pathologies",
                "Mass Lesions",
                "Morphological Alteration",
                "Neurological and muscular",
                "Neurological Disease",
                "Non-structural dysphonia",
                "Organic",
                "Organo-functional",
                "Other vocal diseases",
                "Pathological disorders",
                "Physiological",
                "Structural Dysphonia",
            ]
        ]

    @property
    def database_usage(self):
        return self.__workbook["Database Usage"].iloc[: self.__num_articles][
            [
                "SVD",
                "Voiced",
                "MEEI",
                "FEMH 2018",
                "FEMH 2019",
                "FEMH (other)",
                "Custom",
                "Unclear",
                "PC-GITA",
                "CLP-GPRS",
                "CIEMPIESS",
                "HUPA",
                "DPV",
                "PdA",
                "AVPD",
                "Unnamed public DB",
            ]
        ]

    @property
    def database_characteristics(self):
        return self.__workbook["Database Characteristics"].iloc[: self.__num_articles][
            [
                "Number of channels",
                "Background noise (dB)",
                "Microphone",
                "Audio Interface",
                "Software",
                "Sampling rate; bit depth",
                "Bit depth",
                "Further information source (referenced):",
            ]
        ]

    @property
    def protocol_checklist(self):
        return self.__workbook["Protocol Checklist"].iloc[: self.__num_articles]

    def class_usage(self, min_usage: int):
        class_numbers = self.diagnostic_class_numbers
        class_numbers["counts"] = self.diagnostic_class_counts
        class_numbers = class_numbers.drop(columns=["counts"])
        column_names = class_numbers.columns.tolist()
        class_presence = class_numbers.notna().astype(int)
        dists = sorted(
            [(class_presence[c].sum(), c) for c in column_names], key=lambda x: x[0]
        )
        selected_columns = [x[1] for x in dists if x[0] > min_usage]
        return class_numbers[selected_columns]

    def co_occurence(self, X: pd.DataFrame, Y: pd.DataFrame):
        X = X.stack().notna().reset_index(level=1).rename(columns={"level_1": "X"})
        Y = Y.stack().notna().reset_index(level=1).rename(columns={"level_1": "Y"})
        pairs = X.merge(right=Y, left_index=True, right_index=True)
        co_occurence = pairs.groupby(by=["X", "Y"])["0_x"].count().reset_index()
        return co_occurence.rename(columns={"0_x": "frequency"})
