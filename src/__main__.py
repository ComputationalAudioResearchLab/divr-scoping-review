from class_argparse import ClassArgParser

from .reporter import Reporter


class Main(ClassArgParser):

    def __init__(self) -> None:
        super().__init__(name="BMJ Scoping Review")
        self.__reporter = Reporter()

    def print_extraction_instrument(self):
        self.__reporter.print_extraction_instrument()

    def classification_pipeline(self):
        self.__reporter.classification_pipeline()


if __name__ == "__main__":
    Main()()
