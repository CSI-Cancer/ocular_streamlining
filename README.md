# streamlining training


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


### Environment and Dependency Handling

The repo uses poetry to handle the environment and dpendencies. To install poetry: 
<p><code>curl -sSL https://install.python-poetry.org | python3 -</code></p>

Once the poetry is installed. Go to the root repository and use the below commands.

### Commands
The Makefile contains the central entry points for common tasks related to this project.

#### To create virtual environment and install all the dependencies
<p><code>make install</code></p>

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
