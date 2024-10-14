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

# outcome column prediction
OUTCOME = 'interesting'

# Cohorts for training
COHORT = []

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