from pathlib import Path

from loguru import logger
from tqdm import tqdm
import sys
import pandas as pd
import joblib

from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle
from sklearn.ensemble import HistGradientBoostingClassifier


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
    
def get_data(train_path, val_path, test_path, features_path):
    # Load data
    train_data = pd.read_csv(train_path, converters={'slide_id':str})
    val_data = pd.read_csv(val_path, converters={'slide_id':str})
    test_data = pd.read_csv(test_path, converters={'slide_id':str})
    features = pd.read_csv(features_path, header=None).values.flatten().tolist()
    train_data =  shuffle(train_data, random_state=42)
    X_train = train_data[features]
    y_train = train_data["interesting"]
    X_val = val_data[features]
    y_val = val_data["interesting"]
    X_test = test_data[features]
    y_test = test_data["interesting"]
    # Since we are doing a K cross validation, we will combine the training and validation data
    X_train = pd.concat([X_train, X_val])
    y_train = pd.concat([y_train, y_val])
    return X_train, y_train, X_test, y_test


def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.txt",
    train_path: Path = PROCESSED_DATA_DIR / "train_data.csv",
    val_path: Path = PROCESSED_DATA_DIR / "val_data.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test_data.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    sweep_config: dict = sweep_config
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training...")
    # Load data
    X_train, y_train, X_test, y_test = get_data(train_path,
                                                                val_path,
                                                                test_path,
                                                                features_path)

    # Over-sample the minority in the training data
    sampler = get_oversampler("smote")
    X_train, y_train = sampler.fit_resample(X_train, y_train)

    # Sweep over hyperparameters
    model = HistGradientBoostingClassifier(random_state=42)
    sweep_type = RandomizedSearchCV(estimator=model,
                                    param_distributions=sweep_config,
                                    n_iter=1,
                                    scoring='average_precision',
                                    n_jobs=-1,
                                    cv=2,
                                    random_state=42,
                                    verbose=1)
    # fit gridsearch
    sweep_type.fit(X_train, y_train)
    logger.info(f"Sweeping complete")

    sweep_results=pd.DataFrame.from_dict(sweep_type.cv_results_,orient='columns')
    sweep_results.to_csv(str(MODELS_DIR) + '/search_results.csv')
    
    best_model = sweep_type.best_estimator_
    # save best model for validation
    logger.info("Saving best Model...")
    joblib.dump(best_model, model_path)
    logger.info("saving features...")
    columns_path = MODELS_DIR / "features.txt"
    with open(columns_path, 'w') as f:
        for column in X_train.columns:
            f.write(f"{column}\n")
    logger.info(f"Saved columns of X_train to {columns_path}")
    logger.success("Modeling training complete.")
    # -----------------------------------------

if __name__ == "__main__":
    main()
