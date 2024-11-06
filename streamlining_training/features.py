from pathlib import Path
import sys

from loguru import logger
from tqdm import tqdm

# ML packages
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE


# Visualizations packages
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.utils import shuffle

# Add the project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from streamlining_training.config import (
    PROCESSED_DATA_DIR, THRESH, DROP_FEATURES, model_params,
    scoring_types
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
    
def get_model(model_type, model_params):
    if model_type == "HistGradientBoostingClassifier":
        params = model_params[model_type]
        model =  HistGradientBoostingClassifier(loss=params["loss"],
                                                learning_rate=params["learning_rate"],
                                                max_iter=params["max_iter"],
                                                max_leaf_nodes=params["max_leaf_nodes"],
                                                max_depth=params["max_depth"],
                                                l2_regularization=params["l2_regularization"],
                                                random_state=42,
                                                verbose=1
                                                )
    elif model_type == "RandomForestClassifier":
        params = model_params[model_type]
        model = RandomForestClassifier(n_estimators=params["n_estimators"],
                                       criterion=params["criterion"],
                                       max_depth=params["max_depth"],
                                       min_samples_split=params["min_samples_split"],
                                       min_samples_leaf=params["min_samples_leaf"],
                                       min_weight_fraction_leaf=params["min_weight_fraction_leaf"],
                                       max_features=params["max_features"],
                                       max_leaf_nodes=params["max_leaf_nodes"],
                                       min_impurity_decrease=params["min_impurity_decrease"],
                                       bootstrap=params["bootstrap"],
                                       oob_score=params["oob_score"],
                                       class_weight=params["class_weight"],
                                       ccp_alpha=params["ccp_alpha"],
                                       random_state=42,
                                       verbose=1
                                       )
    else:
        raise ValueError(f"Unknown model: {model_type}")
    
    return model

def select_features(sampler,
                     model,
                       train_df,
                         val_df,
                           scoring_types,
                           drop_features,
                           thresh:float=THRESH
                           ):
    # Shuffle the data to avoid any ordering bias
    train_df =  shuffle(train_df, random_state=42)
    # Filter the column to be used for the training
    keep_columns = train_df.columns
    keep_columns = keep_columns.drop(drop_features) 
    train_df = train_df[keep_columns]
    val_df = val_df[keep_columns]
    # One-hot encode the categorical columns if any
    categorical_columns = train_df.select_dtypes(include=['object']).columns.to_list()
    train_df = pd.get_dummies(train_df, columns=categorical_columns)
    val_df = pd.get_dummies(val_df, columns=categorical_columns)
    # get the features and the target
    X_train = train_df.drop(columns=["interesting"])
    y_train = train_df["interesting"]
    X_val = val_df.drop(columns=["interesting"])
    y_val = val_df["interesting"]

    # Over-sample the training data
    X_train, y_train = sampler.fit_resample(X_train, y_train)
    # Train the model
    model.fit(X_train, y_train)

    perm_scores = permutation_importance(model, X_val, y_val,
                                                    n_repeats=30,
                                                    random_state=42,
                                                    n_jobs=-1,
                                                    scoring=scoring_types
                                                    )
    for score in scoring_types:
        result = perm_scores[score]
        sorted_importance = result.importances_mean.argsort()

        sorted_features=[keep_columns[idx] for idx in sorted_importance]

        importances = pd.DataFrame(
            result.importances[sorted_importance].T,
            columns=sorted_features,
        )
        importances=importances[sorted_features[::-1]]
        selected_features = list(importances.mean()[importances.mean()>thresh].index)
    return importances, selected_features
                                                    

def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR ,
    output_path: Path = PROCESSED_DATA_DIR ,
    model_params: dict = model_params,
    scoring_types: list = scoring_types,
    DROP_FEATURES: list = DROP_FEATURES,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Feature selection from dataset...")
    train_df = pd.read_csv(input_path / "train_data.csv")
    val_df = pd.read_csv(input_path / "val_data.csv")
    train_slides = train_df["slide_id"].unique()
    val_slides = val_df["slide_id"].unique()

    # Over-sampler
    sampler = get_oversampler("smote")
    model = get_model(model_params["model"], model_params)
    importances, features = select_features(sampler,
                     model,
                       train_df,
                         val_df,
                           scoring_types,
                            DROP_FEATURES,
                           )
    # Save the features
    importances.to_csv(output_path / "importances.csv", index=False)
    with open(output_path / "features.txt", "w") as f:
        for feature in features:
            f.write(f"{feature}\n")

    logger.success("Features election complete.")
    # -----------------------------------------


if __name__ == "__main__":
    main()
