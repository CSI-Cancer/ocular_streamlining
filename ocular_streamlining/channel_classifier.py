import os
import pickle

from loguru import logger

import pandas as pd

from csi_images.csi_events import EventArray
from csi_analysis.pipelines.scan import EventClassifier

REPOSITORY_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHANNEL_CLASSIFIERS_PATH = os.path.join(REPOSITORY_PATH, "models", "channel_classifier")


class ChannelClassifier(EventClassifier):
    COLUMN_NAME = "model_classification"
    PREFERRED_ORDER = ["CK", "V", "CD"]
    CHANNELS_TO_COLUMNS = {
        "CD": "CY5",
        "CK": "TRITC",
        "V": "FITC",
    }

    def __init__(
        self,
        stain_name: str,
        version: str,
        save: bool = False,
    ):
        """
        Initialize the streamlining classifier with the given stain name and version.
        :param stain_name:
        :param version:
        :param save:
        """
        self.stain_name = stain_name
        self.save = save
        self.models, self.version = self.load_model(stain_name, version)
        self.channel_order = [
            channel
            for channel in self.PREFERRED_ORDER
            if channel in self.models.keys()
        ]

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.version})"

    @staticmethod
    def load_model(stain_name: str, version: str) -> tuple[dict[str, tuple], str]:
        """
        Get the channel classifier models and associated features for the given stain name and version.
        :param stain_name: the stain type used for the model
        :param version: the version of the model, which should be an ISO date string
        :return: a dict of tuples of channel classifier models and their associated features, or {} if no models are found.
        """
        # Determine the version if needed
        if version == "latest":
            # Get the latest version
            versions = os.listdir(os.path.join(CHANNEL_CLASSIFIERS_PATH, stain_name))
            version = sorted(versions)[-1]
        logger.info(
            f"Loading {version} channel classifier model for {stain_name} stain..."
        )
        # Populate paths
        base_path = os.path.join(CHANNEL_CLASSIFIERS_PATH, stain_name, version)
        model_names = {p for p in os.listdir(base_path)}
        model_names = {p.split(".")[0] for p in model_names if p.endswith(".pickle")}
        # Break out early if no models exist for this stain or stain/version
        if len(model_names) == 0:
            logger.warning(f"No channel classifier models found in {base_path}")
            return {}, version
        # Load the channel classifier model and associated features needed
        models = {}
        for model_name in model_names:
            with open(os.path.join(base_path, f"{model_name}.pickle"), "rb") as file:
                model = pickle.load(file)
            with open(
                os.path.join(base_path, f"{model_name}_columns.txt"), "r"
            ) as file:
                columns = [line.strip() for line in file]
            models[model_name] = (model, columns)
        return models, version

    def classify_events(self, events: EventArray) -> EventArray:
        """
        Apply the channel classifier model to the given events.
        :param events: the EventArray to classify
        :return: the EventArray with populated metadata
        """

        #### Apply Channel Type classifier START
        # check if there is a model loaded in for current staining assay
        logger.info("Applying channel type classifier...")
        # Currently, streamlining only operates on cells.
        # TODO: update to handle other event types
        labels = ["D"] * len(events)
        booleans = {
            self.CHANNELS_TO_COLUMNS[channel]: [False] * len(events) for channel in
            self.channel_order}
        for channel in self.channel_order:
            model, columns = self.models[channel]
            relevant_data = events.features[columns].to_numpy().tolist()
            predictions = model.predict(relevant_data)
            for i, prediction in enumerate(predictions):
                if prediction == 1:
                    labels[i] += f"|{channel}"
                    booleans[self.CHANNELS_TO_COLUMNS[channel]][i] = True
        if self.COLUMN_NAME in events.metadata:
            logger.warning(f'Overwriting events\' existing "{self.COLUMN_NAME}"...')
        events.add_metadata(pd.DataFrame({self.COLUMN_NAME: labels}))
        events.add_metadata(pd.DataFrame(booleans))
        return events
