# streamlining training

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

streamlning training and Pipeline setup

## Project Organization

```
├
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         streamlining_training and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── streamlining_training   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes streamlining_training a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Input file

The list of slides in the excel sheet format is used as the raw immutable data from which canonical data that is input the model is used. Name of the file does not matter as long as the file extension is .xlsx. Typically the excel sheet has two columns. Slide_id and Classification. The classification column is optional. This needs to be manually placed in the ./data/raw/ folder.
<table>
<thead>
<tr>
<th>slide_id</th>
<th>classifications</th>
</tr>
</thead>
<tbody>
<tr>
<td>0B58703</td>
<td>NBD</td>
</tr>
</tbody>
</table>

### make install

### make prepare_data

### make feature_select

### make train

### make evaluate

### make clean_data

### make clean_venv

### make clean_models
