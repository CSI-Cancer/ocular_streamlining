# OCULAR Streamlining: Training & Pipeline

## Getting Started

Clone this repository, as well as
the [CSI-Cancer/csi_analysis](https://github.com/CSI-Cancer/csi_analysis)
and [CSI-Cancer/csi_utils](https://github.com/CSI-Cancer/csi_utils)
to the same directory.

```commandline
git clone git@github.com:CSI-Cancer/ocular_streamlining.git
git clone git@github.com:CSI-Cancer/csi_utils.git
git clone git@github.com:CSI-Cancer/csi_analysis.git
```

Enter the `ocular_streamlining` directory, create a virtual environment, activate and
install the package and all dependencies:

```commandline
cd ocular_streamlining
python -m venv .venv
source .venv/bin/activate
make install
```

If you do not have `poetry` installed, you can install it globally with:

```commandline
sudo apt install pipx
pipx install poetry
```

Or just locally:

```commandline
pip install poetry
```

You should now be able to run scripts and training.

## Project Organization

```
├── Makefile                <- `make` aliases; `make install`, `make data`, etc.
├── README.md               <- You really should read this
├── pyproject.toml          <- Project configuration file with package metadata, 
│                              configurations, and dependencies.
├── requirements.txt        <- Install dependencies manually with 
|                              `pip install -r requirements.txt`
├── data                    <- Directory for storing data
│ ├── external                  <- Data from third party sources
│ ├── interim                   <- Intermediate data that has been transformed
│ ├── processed                 <- The final, canonical data sets for modeling
│ └── raw                       <- The original, immutable data dump
│
├── docs                    <- pdocs generated HTML documentation
|
├── models                  <- Trained models, predictions, or model summaries
|
├── notebooks               <- Jupyter notebooks. Naming convention is a number 
|                              (for ordering), the creator's initials, and a 
|                              short `-` delimited description, e.g. 
|                              `1.0-RMN-initial-data-exploration`
|
├── references              <- Data dictionaries, manuals, etc.

├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│ └── figures                   <- Generated graphics and figures for reporting
│
├── scripts                 <- Pipeline scripts
| └── do_streamlining.py    <- Script to run models with OCULAR outputs.
|
├── ocular_streamlining     <- Model application source code for pipeline use.
| ├── __init__.py
| ├── channel_classifier.py
| └── streamlining_classifier.py
│
└── streamlining_training   <- Training source code, including supportive modules
  ├── __init__.py
  ├── config.py                 <- Store useful variables and configuration
  ├── dataset.py                <- Scripts to download or generate data
  ├── features.py               <- Code to create features for modeling
  ├── plots.py                  <- Code to create visualizations
  └── modeling                  <- Model training and evaluation source code
    ├── __init__.py
    ├── eval.py                     <- Code to test models
    ├── predict.py                  <- Code to infer using trained models
    └── train.py                    <- Code to train models
```

--------

### Input file

The list of slides in the Excel sheet format is used as the raw immutable data from
which canonical data that is input the model is used. Name of the file does not matter
as long as the file extension is .xlsx. Typically, the Excel sheet has two columns.
"Slide_id" and "Classification". The classification column is optional. This needs to be
manually placed in the ./data/raw/ folder.

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

### Commands

The Makefile contains the central entry points for common tasks related to this project:

#### To extract and create interim dataset

<p><code>make prepare_data</code></p>

#### To feature select from the set of features that exists in ocular

<p><code>make prepare_data</code></p>

#### To Train the model

<p><code>make train</code></p>

#### To evaluate the trained model on the test and val dataset

<p><code>make evaluate</code></p>

#### To clean the interimediate and processed canonical data

<p><code>make clean_venv</code></p>

#### To clean up the trained model

<p><code>make clean_models</code></p>
