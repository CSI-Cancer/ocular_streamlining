from pathlib import Path
import sys


from loguru import logger
from tqdm import tqdm

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import itertools
import glob
import json

# Add the project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from csi_utils import databases
from streamlining_training.config import (
    RAW_DATA_DIR, COHORT_PAIR, INTERIM_DATA_DIR, CUSTOM_SPLIT
)

def remove_suffix(slide_id, marker='_'):
    split_res = slide_id.split(marker)
    return split_res[0]

def add_suffix(slide_id, marker='_SL'):
    split_res = slide_id+marker
    return split_res

def generate_friendly_list(items):
    if len(items) > 1:
        sql_list = [(item,) for item in items]
    else:
        sql_list = [(items[0],)]
    return sql_list

def tube_train_test_split(slides,
                           random_state=43,
                             test_size=0.2):
    # Train val split the tubes
    tubes = np.unique([slide[0:-2] for slide in slides])
    train_tubes, val_tubes = train_test_split(tubes,
                                              test_size=test_size,
                                              random_state=random_state)
    train_slides = [slide for slide in slides if slide[0:-2] in train_tubes]
    val_slides = [slide for slide in slides if slide[0:-2] in val_tubes]
    return train_slides, val_slides


def get_events_and_perML(
        slide_ids,
          classification,
            manual_class="Manual",
            both=["per_ml", "events"]):
    ## Get per ML data from the database
    handler = databases.DatabaseHandler(is_production=True,
                                        username="reader",
                                        write_protect=True)
    # Query for tube ids for all the slides
    slide_ids = generate_friendly_list(slide_ids["slide_id"].apply(remove_suffix).values.tolist())

    data_from_db = {key: None for key in both}

    if "per_ml" in both:
        # Query for slide and tube
        query = ["select slide_id, tube_id from slide where slide_id = (%s)"]*len(slide_ids)
        query_result = handler.get(query, slide_ids)
        slides_tubes = pd.DataFrame(list(itertools.chain.from_iterable(query_result['results'])),
                                columns=query_result['headers'][0] )
        
        # Query for per ML details using tube ids
        tube_values = generate_friendly_list(slides_tubes["tube_id"].values.tolist())
        query = ["select tube_id, wbc_count, patient_id from tube where tube_id = (%s)"]*len(tube_values)
        query_result = handler.get(query, tube_values)
        tubes_wbc_counts = pd.DataFrame(list(itertools.chain.from_iterable(query_result['results'])),
                                columns=query_result['headers'][0] )
        
        # Query analysis table for dapi counts per slide
        query = ["select slide_id, dapi_count_ocular from analysis where slide_id = (%s)"]*len(slide_ids)
        query_result = handler.get(query, slide_ids)
        dapi_counts = pd.DataFrame(list(itertools.chain.from_iterable(query_result['results'])),
                                columns=query_result['headers'][0] )
        per_ml = pd.merge(pd.merge(slides_tubes, tubes_wbc_counts, on="tube_id"),
                        dapi_counts, on="slide_id")
        per_ml["wbc_count"] = per_ml["wbc_count"] * 10**6
        logger.info("")
        not_in_db = list(set([item[0] for item in slide_ids]) - set(per_ml["slide_id"].values))

        if not_in_db:
            logger.info(f"slides not found in the database: {not_in_db}")
        
        data_from_db["per_ml"] = per_ml
    
    if "events" in both:
        ## Get the manual events from the database
        if manual_class == "Manual":
            query = [(
                "select * from ocular_hitlist where slide_id in "
                "(%s) and manual_classification is not NULL"
                " and LENGTH(manual_classification) > 0 and"
                " type not in ('Not interesting', 'common_cell');"
            )]*len(slide_ids)
            query_result = handler.get(query, slide_ids)
            hitlist = pd.DataFrame(list(itertools.chain.from_iterable(query_result['results'])),
                                columns=query_result['headers'][0] )
            
        elif manual_class == "Mixed":
            query = [(
                "select * from ocular_hitlist where slide_id in "
                "(%s) and "
                "type not in ('Not interesting', 'common_cell');"
            )]*len(slide_ids)
            query_result = handler.get(query, slide_ids)
            hitlist = pd.DataFrame(list(itertools.chain.from_iterable(query_result['results'])),
                                columns=query_result['headers'][0] )
        elif manual_class == "Not_interesting":
            query = [(
                "select * from ocular_hitlist where slide_id in "
                "(%s) and "
                " type in ('Not interesting');"

            )]*len(slide_ids)
            query_result = handler.get(query, slide_ids)
            hitlist = pd.DataFrame(list(itertools.chain.from_iterable(query_result['results'])),
                                columns=query_result['headers'][0] )
        else:
            hitlist = pd.DataFrame()
            logger.error("Invalid manual classification")
            sys.exit(1)
        
        if len(hitlist) == 0:
            logger.info(f"No manual events found for {classification}")
        
        hitlist['manual_classification'].replace(np.nan, '', inplace=True)

        no_events = list(set([item[0] for item in slide_ids]) - set(np.unique(hitlist["slide_id"].values)))

        if no_events:
            logger.info(f"No events with classification: {classification} found for the following slides: {no_events}")
        
        hitlist["slide_id"] = hitlist["slide_id"].apply(remove_suffix)

        data_from_db["events"] = hitlist

    return data_from_db

def get_manual_events(slides_cohort, classes):

    # Retrieve manual events and per ML data from the database
    per_ml_df = []
    events_df = []
    for classification in classes:
        slides_ids = slides_cohort[
            slides_cohort["classification"] == classification
            ]
        # Get interesting events
        data_interesting = get_events_and_perML(slides_ids,
                                                 classification,
                                                 manual_class="Mixed",)
        events = data_interesting["events"]
        per_ml = data_interesting["per_ml"]

        events['interesting']=1
        events_df.append(events)

        logger.info(f"Getting per ml and events for {classification}")
        data_not_interesting = get_events_and_perML(slide_ids=slides_ids,
                                                    classification=classification,
                                                    manual_class="Not_interesting",
                                                    both=["events"])
        events_not_interesting = data_not_interesting["events"]
        events_not_interesting['interesting']=0
        events_df.append(events_not_interesting)

        per_ml["classification"] = classification
        per_ml_df.append(per_ml)

    # concat the dataframes
    events = pd.concat(events_df)
    per_ml = pd.concat(per_ml_df)
    return per_ml, events

def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR ,
    output_path: Path = INTERIM_DATA_DIR,
    cohort: list = COHORT_PAIR,
    split: dict = CUSTOM_SPLIT,
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Looking for the list of slides")
    slides_path = glob.glob(str(input_path / "*.xlsx"))
    if len(slides_path) > 0:
        if cohort:
            slides_cohort = pd.read_excel(slides_path[0])
            slides_cohort = slides_cohort[slides_cohort["classification"].isin(cohort)]
            classes = cohort
        else:
            slides_cohort = pd.read_excel(slides_path[0])
            classes = slides_cohort["classification"].unique()
        
        
        logger.info(f"Classes found: {classes} \t Number of slides: {len(slides_cohort)}")
    else:
        logger.error("No file with list of slides found.")
        sys.exit(1)

    # Get manual events and per ML data from DB
    per_ml, events = get_manual_events(slides_cohort, classes)
    logger.info("Manual and Per ML data retrieved from the database.")
    events = events.drop(columns=['data'], axis=1)
    events["slide_id"] = events["slide_id"].apply(add_suffix)
    per_ml["slide_id"] = per_ml["slide_id"].apply(add_suffix)
    logger.info(f"Saving data to excel files in {output_path} folder.")
    events.to_excel(output_path / "events.xlsx", index=False)
    per_ml.to_excel(output_path / "per_ml.xlsx", index=False)

    slides_cohort = list(set(per_ml["slide_id"]))

    if split:
        train_slides = per_ml[(
            per_ml['classification'].isin(split['train'])
            )]['slide_id'].to_list()
        test_slides = per_ml[(
            per_ml['classification'].isin(split['test'])
            )]['slide_id'].to_list()
    else:
        train_slides, test_slides = tube_train_test_split(slides_cohort,
                                                          random_state=43,
                                                          test_size=0.2)
    
    train_slides, val_slides = tube_train_test_split(train_slides,
                                                     random_state=43,
                                                     test_size=0.2)
    # Save the train, val and test slides
    with open(f'{output_path}/train_slides.json', 'w') as f:
        json.dump(train_slides, f)
    with open(f'{output_path}/val_slides.json', 'w') as f:
        json.dump(val_slides, f)
    with open(f'{output_path}/test_slides.json', 'w') as f:
        json.dump(test_slides, f)
    
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    main()
