from pathlib import Path
import sys

from loguru import logger
from tqdm import tqdm

# ML packages
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.inspection import permutation_importance

# Visualizations packages
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# Add the project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from streamlining_training.config import (
    PROCESSED_DATA_DIR, THRESH, DROP_FEATURES,
)



def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR ,
    output_path: Path = PROCESSED_DATA_DIR ,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")

    
    
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    main()
