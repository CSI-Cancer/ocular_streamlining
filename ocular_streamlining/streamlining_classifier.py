import os
import joblib

from loguru import logger

import numpy as np
import pandas as pd

from csi_images.csi_events import EventArray
from csi_analysis.pipelines.scan_pipeline import EventClassifier

REPOSITORY_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STREAMLINING_MODELS_PATH = os.path.join(REPOSITORY_PATH, "models", "streamlining")


class StreamliningClassifier(EventClassifier):
    COLUMN_NAME = "ocular_interesting"

    def __init__(
        self,
        stain_name: str,
        version: str,
        threshold: float = 0.5,
        save: bool = False,
    ):
        """
        Initialize the streamlining classifier with the given stain name and version.
        :param stain_name:
        :param version:
        :param threshold:
        :param save:
        """
        self.stain_name = stain_name
        self.threshold = threshold
        self.save = save
        self.model, self.columns, self.version = self.load_model(stain_name, version)

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version})"

    @staticmethod
    def load_model(stain_name: str, version: str):
        """
        Get the streamlining model and associated columns for the given stain name and version.
        :param stain_name: the stain type used for the model
        :param version: the version of the model, which should be an ISO date string
        :return: a tuple of the joblib model, list of column names, and version string
        """
        # Determine the version if needed
        if version == "latest":
            # Get the latest version
            versions = os.listdir(os.path.join(STREAMLINING_MODELS_PATH, stain_name))
            version = sorted(versions)[-1]
        logger.info(f"Loading {version} 'interesting' model for {stain_name} stain...")
        # Populate paths
        base_path = os.path.join(STREAMLINING_MODELS_PATH, stain_name, version)
        model_path = os.path.join(base_path, f"{version}.joblib")
        columns_path = os.path.join(base_path, f"{version}.txt")
        # Load the "interesting" classification model and associated features needed
        interesting_classifier = joblib.load(model_path)
        with open(columns_path, "r") as file:
            streamlining_columns = [line.strip() for line in file]
        return interesting_classifier, streamlining_columns, version

    def predict_proba(self, events: EventArray):
        """
        Apply the interesting classifier model to the given events.
        :param events: the EventArray to classify
        :return: np.ndarray of the model output probability
        """
        logger.info("Applying interesting classifier...")
        # Apply streamlining classifier based on threshold
        streamlining_data = events.features[self.columns]
        probabilities = self.model.predict_proba(streamlining_data)[:, 1]
        return probabilities

    def classify_events(
        self, events: EventArray, probabilities: np.ndarray = None
    ) -> EventArray:
        """
        Classify the given events using the streamlining classifier.
        :param events: the EventArray to classify
        :param probabilities: an optionally pre-populated probabilities array
        :return: the EventArray with populated metadata
        """
        # Apply the interesting classifier to the events
        if probabilities is None:
            probabilities = self.predict_proba(events)
        predictions = np.where(probabilities > self.threshold, 1, 0)
        # Add the classification to the metadata
        if self.COLUMN_NAME in events.metadata:
            logger.warning(f'Overwriting events\' existing "{self.COLUMN_NAME}"...')
        events.add_metadata(pd.DataFrame({self.COLUMN_NAME: predictions}))
        return events
