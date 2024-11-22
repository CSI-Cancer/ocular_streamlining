#!/usr/bin/env python

import os
import sys
import shutil
import argparse

import pandas as pd

from loguru import logger

from csi_utils import csi_databases, csi_paths
from csi_images import ocular_files
from csi_images.csi_events import EventArray
from csi_analysis.modules.ocular_report_clusterer import OcularReportClusterer

from ocular_streamlining.channel_classifier import ChannelClassifier
from ocular_streamlining.streamlining_classifier import StreamliningClassifier

# OCULAR files primarily used in streamlining
TARGETED_OCULAR_FILES = [
    "rc-final1.rds",
    "rc-final2.rds",
    "rc-final3.rds",
    "rc-final4.rds",
    "ocular_interesting.rds",
]

# Create backup of target files if first SL run
# Read from backup files if they exist
BACKUP_OCULAR_FILES = TARGETED_OCULAR_FILES + [
    "ocular_interesting.rds",
    "ocular_not_interesting.rds",
    "rc-final.csv",
    "rc-final.rds",
    "rc-final1.rds",
    "rc-final2.rds",
    "rc-final3.rds",
    "rc-final4.rds",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Streamlining, an added step in OCULAR that "
        "improves classification and eases human curation."
    )

    parser.add_argument("slide_id", type=str, help="Target slide ID")
    parser.add_argument(
        "--version",
        type=str,
        default="latest",
        help="Model version to use for streamlining; defaults to latest",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Threshold for interesting classification",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Run the streamlining in production mode, using the production database.",
    )
    parser.add_argument(
        "--development",
        action="store_true",
        help="Run the streamlining in development mode, using the development database.",
    )
    parser.add_argument(
        "--stain_name",
        type=str,
        default=None,
        help="Name of the staining protocol used on the slide. "
        "Use for local runs and in lieu of querying the database.",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=None,
        help="Path to the report folder, where OCULAR files are stored. "
        "The same path is used for input and output, "
        "although the program backs up the files it overwrites."
        "Defaults to the production or development path if selected.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # Sanitize arguments
    args.slide_id = args.slide_id.upper().strip()
    if len(args.slide_id) < 6 or len(args.slide_id) >= 32:
        raise ValueError(f"Slide ID is not the right size: {args.slide_id}")
    for c in args.slide_id:
        if not c.isalnum() and c != "_":
            raise ValueError(f"Slide ID {args.slide_id} cannot contain character '{c}'")

    if not args.production and not args.development:
        if args.stain_name is None:
            raise ValueError(
                "Must be in production or development mode, or provide a stain name"
            )
        elif args.path is None:
            raise ValueError(
                "Must be in production or development mode, "
                "or provide a path to the report folder"
            )
    if args.production and args.development:
        raise ValueError("Cannot be in both production and development modes")

    return parser.parse_args()


def backup_files(
    report_path,
    backup_folder_name: str = "ocular_clust_rc",
    files: list[str] = (
        "ocular_interesting.rds",
        "ocular_not_interesting.rds",
        "rc-final.csv",
        "rc-final.rds",
        "rc-final1.rds",
        "rc-final2.rds",
        "rc-final3.rds",
        "rc-final4.rds",
    ),
):
    backup_path = os.path.join(report_path, backup_folder_name)
    if not os.path.exists(backup_path):
        os.makedirs(backup_path)
    for file in files:
        src = os.path.join(report_path, file)
        dst = os.path.join(backup_path, file)
        if os.path.isfile(src):
            if not os.path.isfile(dst):  # Don't overwrite backups
                shutil.copyfile(src, dst)
        else:
            logger.warning(f"Can't backup {file}; file not found")
            if file not in [
                "ocular_interesting.rds",
                "ocular_not_interesting.rds",
            ]:
                raise ValueError(f"Required file {file} not found")


def main():
    args = parse_arguments()

    slide_id = args.slide_id
    version = args.version
    threshold = args.threshold

    if args.production:
        report_path = os.path.join(csi_paths.REPORTS["production"], slide_id, "ocular")
        handler = csi_databases.DatabaseHandler("reader", is_production=True)
    elif args.development:
        report_path = os.path.join(csi_paths.REPORTS["development"], slide_id, "ocular")
        handler = csi_databases.DatabaseHandler("reader", is_production=False)
    else:
        report_path = args.path
        handler = None

    # Manual path override
    if args.path is not None:
        report_path = args.path

    # Set up logger
    level = "DEBUG" if args.verbose else "INFO"
    log_path = os.path.join(report_path, "log", "2b-streamlining.log")
    logger.remove(0)
    logger.add(sys.stderr, level=level, colorize=True)
    logger.add(log_path, level=level, backtrace=True, catch=True, mode="a")
    logger.info("================================================================")
    logger.info(f"Starting streamlining for slide {slide_id}")

    # Derive the staining assay name
    if args.stain_name is not None:
        # Manually passed stain name takes precedence
        stain_name = args.stain_name
    elif handler is not None:
        # Otherwise, we need to query the database
        stain_name = csi_databases.query_stain_name(handler, slide_id)
    else:
        raise ValueError("Stain name not provided and options were not set for queries")

    # Back up key files from OCULAR clustering & Atlas
    backup_files(report_path)

    # Load OCULAR data and set our version
    events = EventArray.load_ocular(report_path)

    # Add frame-level statistics and morphometric data
    logger.info("Adding frame-level statistics and morphometric data...")
    frame_stats = ocular_files.get_frame_statistics(report_path)
    slide_stats = ocular_files.get_slide_statistics(report_path)
    events_with_stats = ocular_files.merge_statistics(events, frame_stats, slide_stats)
    events_with_stats = ocular_files.filter_and_generate_statistics(events_with_stats)

    # Update channel classifications if a classifier is available
    channel_classifier = ChannelClassifier(stain_name, version)
    if channel_classifier.models != {}:
        events_with_stats = channel_classifier.classify_events(events_with_stats)
        events.add_metadata(events_with_stats.metadata)

    # Load the interesting/streamlining classifier model and associated columns
    streamlining_classifier = StreamliningClassifier(stain_name, version, threshold)
    # Update the version in the metadata
    events.metadata["atlas_version"] = f"atlas/{streamlining_classifier.version}"

    probabilities = streamlining_classifier.predict_proba(events_with_stats)
    events_with_stats = streamlining_classifier.classify_events(
        events_with_stats, probabilities
    )
    # Add the interesting classification to the metadata
    events.add_metadata(events_with_stats.metadata[streamlining_classifier.COLUMN_NAME])

    # Regular data needs to get re-clustered for the report
    interesting_rows = events.metadata[streamlining_classifier.COLUMN_NAME]
    interesting_rows = interesting_rows.to_numpy().astype(bool)
    data = events_with_stats.rows(~interesting_rows)
    logger.info(f"{sum(interesting_rows)} events classified as interesting by streamlining")

    logger.info(f"Re-clustering {len(data)} events not predicted as interesting...")
    data.add_metadata(pd.DataFrame({"TEMP_P": probabilities[~interesting_rows]}))
    clusterer = OcularReportClusterer(
        columns=streamlining_classifier.columns,
        sort_by="TEMP_P",
        ascending=False,
    )
    data = clusterer.classify_events(data)
    events.metadata["clust"] = events.metadata["unique_id"].replace(
        to_replace=data.get("unique_id").to_numpy().tolist(),
        value=data.get("cluster_id").to_numpy().tolist(),)

    events.save_ocular(report_path)
    logger.info("Saved results to OCULAR files for reporting.")


if __name__ == "__main__":
    main()
