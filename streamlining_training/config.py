from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import platform
import os
import sys

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Local data storage location
if PROJ_ROOT:
    os.makedirs(PROJ_ROOT/"data", exist_ok=True)
    DATA_DIR = PROJ_ROOT / "data"
else:
    logger.error("PROJ_ROOT is not defined")
    sys.exit(1)

# Network data storage location
if platform.system() == "Windows":
    NETWORK_ROOT = "\\\\csi-nas.usc.edu"
elif platform.system() == "Linux":
    NETWORK_ROOT = "/mnt"
else:
    logger.info("Unknown operating system")

# production and devlopment data storage location
PRODUCTION_REPORT_DIR = Path(NETWORK_ROOT) / "HDSCA_Pipeline" / "DZ"
PRODUCTION_SECRET_PATH = Path(NETWORK_ROOT) / "secrets"
DEVELOPMENT_REPORT_DIR = Path(NETWORK_ROOT) / "HDSCA_Development" / "DZ"
DEVELOPMENT_SECRET_PATH = Path(NETWORK_ROOT) / "HDSCA_Development" / "secrets"


# Local repo data storage location
RAW_DATA_DIR = DATA_DIR / "raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)
INTERIM_DATA_DIR = DATA_DIR / "interim"
os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
PROCESSED_DATA_DIR = DATA_DIR / "processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
EXTERNAL_DATA_DIR = DATA_DIR / "external"
os.makedirs(EXTERNAL_DATA_DIR, exist_ok=True)


# Model storage location
MODELS_DIR = PROJ_ROOT / "models"

# Analsysis and reports
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# Training related parameters
# TODO: Training config needs to be changed to be acceptable from YAML file

# outcome column prediction
OUTCOME = 'interesting'

# Cohorts for training
COHORT_PAIR = []

CUSTOM_SPLIT = {}

# performance threshold for filtering based on feature importance
THRESH = 0.0

# drop features
DROP_FEATURES = [OUTCOME] + ['slide_id',
                    'model_classification',
                    'frame_id','cell_id',
                    'unique_id',
                    'cellx','celly','catalogue_id','catalogue_classification',
                    'catalogue_distance',
                    'cellcluster_id',
                    'cellcluster_count',
                    'framegroup',
                    'rowname',
                    'atlas_version',
                    'TRITC',
                    'CY5',
                    'FITC',
                    'clust',
                    'hcpc',
                    'slide_sdev_level_0',
                    'slide_sdev_index',
                    'not_interesting_report_tab',
                    'Unnamed',
                    'Unnamed: 0']
DROP_FEATURES += ['hcpc', 'atlas_id',
                     'atlas_distance', 'atlas_version']

# feature engineering
SELECT_FEATURES = True

# Detailed evaluation
TARGET_SLIDES = 'test'
VISUALIZE_EVENTS = {0.5:True}


# OCULAR parameters
drop_dapi_neg = True
drop_channel_class = ["(DAPI-)CD", "(Dapi-)CD", "(Dapi-)V", "(Dapi-)V|CD"]

ocular_params = {
    'not_int_sample_size': 1000,
    'drop_columns': True,
    'drop_common_cells': True,
    'dapipos': True,
    'drop_morphs': ['theta', 'blurred','haralick','multi_channel',
                    'remove_slide_level', 'remove_frame_level'],
    'keep_morphs': [],
    'dapi_pos_files': ["rc-final1.rds",
        "rc-final2.rds",
        "rc-final3.rds",
        "rc-final4.rds",
        "ocular_interesting.rds"],
    'dapi_neg_files': ["others-final1.rds",
        "others-final2.rds",
        "others-final3.rds",
        "others-final4.rds"],
    
}

# Data preprocessing parameters
channel_map = {'D' :'dapi',
               'CK' : 'tritc',
               'V' : 'fitc',
               'CD' : 'cy5'
               }

feature_sdom_levels = {
    'frame': ['cellf.tritc.b.mean', 'cellf.tritc.b.sd', 'cellf.tritc.b.mad',
              'cellf.tritc.b.q001', 'cellf.tritc.b.q005', 'cellf.tritc.b.q05',
              'cellf.tritc.b.q095', 'cellf.tritc.b.q099', 'tritc_cy5_ratio',

              'cellf.fitc.b.mean', 'cellf.fitc.b.sd', 'cellf.fitc.b.mad',
              'cellf.fitc.b.q001', 'cellf.fitc.b.q005', 'cellf.fitc.b.q05',
              'cellf.fitc.b.q095', 'cellf.fitc.b.q099', 
              
              'cellf.cy5.b.mean', 'cellf.cy5.b.sd', 'cellf.cy5.b.mad',
              'cellf.cy5.b.q001', 'cellf.cy5.b.q005', 'cellf.cy5.b.q05',
              'cellf.cy5.b.q095', 'cellf.cy5.b.q099',
              
              'nucleusf.dapi.b.mean', 'nucleusf.dapi.b.sd', 'nucleusf.dapi.b.mad',
              'nucleusf.dapi.b.q001', 'nucleusf.dapi.b.q005', 'nucleusf.dapi.b.q05',
              'nucleusf.dapi.b.q095', 'nucleusf.dapi.b.q099'],
}

event_id = ['frame_id', 'slide_id', 'cell_id', 'cellx', 'celly']