from pathlib import Path


from loguru import logger
from tqdm import tqdm
import sys
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler


# Add the project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    MODELS_DIR, PROCESSED_DATA_DIR, sweep_config
)


def get_oversampler(sampler="random"):
    if sampler == "random":
        return RandomOverSampler(sampling_strategy="minority",
                                  random_state=42)
    elif sampler == "smote":
        return SMOTE(sampling_strategy="minority",
                     random_state=42)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")


def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.txt",
    train_path: Path = PROCESSED_DATA_DIR / "train_data.csv",
    val_path: Path = PROCESSED_DATA_DIR / "val_data.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test_data.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    sweep_config: dict = sweep_config
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training...")
    # Load data
    train_data = pd.read_csv(train_path, converters={'slide_id':str})
    val_data = pd.read_csv(val_path, converters={'slide_id':str})
    test_data = pd.read_csv(test_path, converters={'slide_id':str})
    features = pd.read_csv(features_path, header=None).values.flatten().tolist()
    X_train = train_data[features]
    y_train = train_data["interesting"]
    X_val = val_data[features]
    y_val = val_data["interesting"]

    # Over-sample the minority in the training data
    sampler = get_oversampler("smote")
    X_train, y_train = sampler.fit_resample(X_train, y_train)

    # Sweep over hyperparameters
    
    logger.success("Modeling training complete.")
    # -----------------------------------------

if __name__ == "__main__":
    main()
