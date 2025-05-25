# DiVR (Disordered Voice Recognition) - Scoping Review

This repository contains analysis and figure generation tools built for the scoping review. This repository is made public for two reasons:

1. To provide access to the [full extraction instrument](./data/Scoping%20Review%20Extraction.xlsx)
2. To provide access to the code used for image generation and analysis

## How to cite

Coming Soon

In the meantime you can cite our [protocol](https://bmjopen.bmj.com/content/14/2/e076998.long)

```
@article {divr_protocl,
	author = {Gupta, Rijul and Gunjawate, Dhanshree R and Nguyen, Duy Duong and Jin, Craig and Madill, Catherine},
	title = {Voice disorder recognition using machine learning: a scoping review protocol},
	volume = {14},
	number = {2},
	elocation-id = {e076998},
	year = {2024},
	doi = {10.1136/bmjopen-2023-076998},
	publisher = {British Medical Journal Publishing Group},
	abstract = {Introduction Over the past decade, several machine learning (ML) algorithms have been investigated to assess their efficacy in detecting voice disorders. Literature indicates that ML algorithms can detect voice disorders with high accuracy. This suggests that ML has the potential to assist clinicians in the analysis and treatment outcome evaluation of voice disorders. However, despite numerous research studies, none of the algorithms have been sufficiently reliable to be used in clinical settings. Through this review, we aim to identify critical issues that have inhibited the use of ML algorithms in clinical settings by identifying standard audio tasks, acoustic features, processing algorithms and environmental factors that affect the efficacy of those algorithms.Methods We will search the following databases: Web of Science, Scopus, Compendex, CINAHL, Medline, IEEE Explore and Embase. Our search strategy has been developed with the assistance of the university library staff to accommodate the different syntactical requirements. The literature search will include the period between 2013 and 2023, and will be confined to articles published in English. We will exclude editorials, ongoing studies and working papers. The selection, extraction and analysis of the search data will be conducted using the {\textquoteleft}Preferred Reporting Items for Systematic Reviews and Meta-Analyses extension for scoping reviews{\textquoteright} system. The same system will also be used for the synthesis of the results.Ethics and dissemination This scoping review does not require ethics approval as the review solely consists of peer-reviewed publications. The findings will be presented in peer-reviewed publications related to voice pathology.},
	issn = {2044-6055},
	URL = {https://bmjopen.bmj.com/content/14/2/e076998},
	eprint = {https://bmjopen.bmj.com/content/14/2/e076998.full.pdf},
	journal = {BMJ Open}
}
```

## How to use

This repo is not designed to be used as a library currently, but if are keen to explore it then you can access the various analysis tools using the python module

```bash
python -m src --help

usage: BMJ Scoping Review [-h] {accuracy_by_diag,classification_circles,classification_circles_2,classification_labels,classification_pipeline,data_balancing,pipeline_per_label,print_extraction_instrument} ...

positional arguments:
  {accuracy_by_diag,classification_circles,classification_circles_2,classification_labels,classification_pipeline,data_balancing,pipeline_per_label,print_extraction_instrument}

options:
  -h, --help            show this help message and exit
```

## How to develop

The repo is setup using VSCode devcontainers, so you can follow instructions to set up devcontainers and then open this repo in a devcontainer which should set everything up automatically.

If you don't wish to use devcontainer, then you can simply activate a pipenv shell using the provided [Pipfile](./Pipfile).
