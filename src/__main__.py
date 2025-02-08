from pathlib import Path
from class_argparse import ClassArgParser

from .reporter import Reporter


class Main(ClassArgParser):

    def __init__(self) -> None:
        super().__init__(name="BMJ Scoping Review")
        curdir = Path(__file__).parent.resolve()
        data_file_path = Path(f"{curdir}/../data/Scoping Review Extraction.xlsx")
        figures_path = Path(f"{curdir}/../figures")
        self.__reporter = Reporter(
            data_file_path=data_file_path, figures_path=figures_path
        )

    def print_extraction_instrument(self):
        self.__reporter.print_extraction_instrument()

    def classification_labels(self):
        self.__reporter.classification_labels()

    def classification_pipeline(self):
        self.__reporter.classification_pipeline()

    def classification_circles_2(self):
        self.__reporter.classification_circles_2()

    def classification_circles(self):
        self.__reporter.classification_circles()


if __name__ == "__main__":
    Main()()
