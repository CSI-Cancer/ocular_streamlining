from pathlib import Path

from loguru import logger
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    MODELS_DIR, PROCESSED_DATA_DIR, sweep_config, thresh_search
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import average_precision_score, accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, f1_score

import json

def get_data(val_path, test_path, features_path):
     # Load data
    test_data = pd.read_csv(test_path, converters={'slide_id':str})
    val_data = pd.read_csv(val_path, converters={'slide_id':str})
    features = pd.read_csv(features_path, header=None).values.flatten().tolist()
    X_test = test_data[features]
    X_val = val_data[features]
    y_test = test_data["interesting"]
    y_val = val_data["interesting"]
    return X_val, y_val, X_test, y_test, 

def score_model(clf, X_test, y_test, thresh):
    Y_pred_prob = clf.predict_proba(X_test)
    y_true = y_test.copy().to_numpy()
    y_pred = np.where(Y_pred_prob[:, 1] > thresh, 1, 0)

    f1 = f1_score(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)

    conf_matrix = confusion_matrix(y_true, y_pred)

    if (y_true==y_pred).all() and len(conf_matrix)==1:
        false_negatives=0
        false_positives=0
        if sum(y_pred)==0:
            true_negatives=len(y_pred)
            true_positives=0
        elif sum(y_pred)>0:
            true_negatives=0
            true_positives=len(y_pred)

        conf_matrix = np.array([[true_negatives,false_positives],
                             [false_negatives, true_positives]])
    else:
        true_negatives = conf_matrix[0, 0]
        false_positives = conf_matrix[0, 1]
        true_positives = conf_matrix[1, 1]
        false_negatives = conf_matrix[1, 0]
    
    if true_negatives+false_positives==0:
        specificity=1
    else:
        specificity = true_negatives / (true_negatives + false_positives)
    # Calculate Precision
    precision = precision_score(y_true, y_pred,zero_division=1.0)
    
    # Calculate Sensitivity (Recall)
    recall = recall_score(y_true, y_pred, zero_division=1.0)
    
    accuracy=accuracy_score(y_true, y_pred)

    return {"f1":f1,
            "avg_precision":avg_precision,
            "specificity":specificity,
            "precision":precision,
            "recall":recall,
            "accuracy":accuracy,
            "false_negatives":str(false_negatives),
            "false_positives":str(false_positives),
            "true_negatives":str(true_negatives),
            "true_positives":str(true_positives)}

def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.txt",
    val_path: Path = PROCESSED_DATA_DIR / "val_data.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test_data.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    thresh: list = thresh_search
    # -----------------------------------------
):
    try:
        clf = joblib.load(model_path)
    except:
        logger.error(f"Model not found at {model_path}.")
        sys.exit(1)

    X_val, y_val, X_test, y_test = get_data(val_path, test_path, features_path)
    model_metrics = {"val":{}, "test":{}}
    for data in model_metrics.keys():
        if data == "val":
            for th in thresh:
                score = score_model(clf, X_val, y_val, th)
                model_metrics[data][str(th)] = score
        else:
            for th in thresh:
                score = score_model(clf, X_test, y_test, th)
                model_metrics[data][str(th)] = score
    
    with open(MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump(model_metrics, f)
    logger.success("Model evaluation complete.")        

if __name__ == "__main__":
    main()