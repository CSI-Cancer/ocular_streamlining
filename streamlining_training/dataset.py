from pathlib import Path
import sys


from loguru import logger
from tqdm import tqdm
import pyreadr

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import itertools
import glob
import json
from itertools import permutations

# Add the project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from csi_utils import databases
from streamlining_training.config import (
    RAW_DATA_DIR, COHORT_PAIR, INTERIM_DATA_DIR, CUSTOM_SPLIT, 
    PRODUCTION_REPORT_DIR, PROCESSED_DATA_DIR, ocular_params,
    channel_map, feature_sdom_levels
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
        query = ["select slide_id, tube_id from slide where slide_id = %s"]*len(slide_ids)
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

def get_data_from_files(slide_id: str,
                         ocular_params: dict,
                         ocular_path: Path = PRODUCTION_REPORT_DIR):  
    file_holder = {}
    if ocular_params["dapipos"]:
        event_files = ocular_params["dapi_pos_files"]
    else:
        event_files = ocular_params["dapi_neg_files"]
    
    failed_loads = []
    empty_files = []

    for file in event_files:
        file_path = ocular_path / slide_id / "ocular" / file

        try:
            file_holder[file] = pyreadr.read_r(file_path)
        except Exception as e:
            logger.error(f"Traceback while loading files for the slide : {slide_id} \n {e}")
            failed_loads.append(file)
            continue
        file_holder[file] = file_holder[file][None]
        if (file == "ocular_interesting.rds" or file == "ocular_not_interesting.rds"
            ) and ocular_params["drop_common_cells"]:
            file_holder[file] = file_holder[file][
                file_holder[file]["catalogue_classification"] != "common_cell"]
        if len(file_holder[file]) == 0:
            empty_files.append(file)
        else:
            if ocular_params["dapipos"]:
                file_holder[file]["rowname"] = file_holder[file]["rowname"].astype(str)
                split_res = file_holder[file]["rowname"].str.split(" ", n=1, expand=True)

                if len(split_res.columns) != 2:
                        logger.error(f"Slide {slide_id} rowname column has issues")
                file_holder[file][["frame_id", "cell_id"]] = split_res
                file_holder[file][
                        ["frame_id", "cell_id"]
                ] = file_holder[file][["frame_id", "cell_id"]].astype(
                    "int"
                )
            file_holder[file].reset_index(drop=True, inplace=True)

    for empty in empty_files:
        logger.info(f"deleting empty file: {empty} for the slide: {slide_id}")
        del file_holder[empty]
    
    for dailed in failed_loads:
        logger.error(f"Deleting failed to load file: {failed_loads} for the slide: {slide_id}")
        del file_holder[failed_loads]
    
    ocular_data = [
            file_holder[filename]
            for filename in file_holder.keys()
        ]
    if len(ocular_data) != 0:
        ocular_data = pd.concat(ocular_data)
        logger.info(f"Found {len(ocular_data)} events for slide {slide_id} in ocular files")

    return ocular_data

def get_frame_morphs(slide_id, ocular_params,
                     path: Path = PRODUCTION_REPORT_DIR):

    slide__path = path / slide_id / "ocular" 
    morph_files = ["framestat-means.csv", "framestat-dev.csv"]
    prefixes = ["means", "dev"]
    files = {}
    for file, prefix in zip(morph_files, prefixes):
        file_path = slide__path / file
        df = pd.read_csv(file_path)
        df = df.rename(columns={df.columns[0]: 'frame_id'})
        df = df.rename(columns={col: f"frame_{prefix}_{col}" for col in df.columns[1:]})
        files[prefix] = df

    frame_stats = pd.merge(files["means"], files["dev"], on="frame_id")
    return frame_stats

def get_slide_stats(slide_id, ocular_params,
                    path: Path = PRODUCTION_REPORT_DIR):
    slide__path = path / slide_id / "ocular"
    slide_stat_df = pd.read_csv(slide__path / "slidestat-calc.csv")
    # Index 0 is the standard deviation
    slide_dev = slide_stat_df.iloc[1].to_frame().transpose()
    slide_dev.reset_index(inplace=True)
    # Index 0 is the mean
    slide_mean = slide_stat_df.iloc[0].to_frame().transpose()
    slide_mean.reset_index(inplace=True)
    slide_dev.drop(columns=["Unnamed: 0"], inplace=True)
    slide_mean.drop(columns=["Unnamed: 0"], inplace=True)
    slide_dev = slide_dev.add_prefix("slide_dev_")
    slide_mean = slide_mean.add_prefix("slide_means_")\
    
    slide_stats = pd.concat([slide_mean, slide_dev], axis=1)
    return slide_stats

def calculate_sdom(df, feature, prefix):
    feature_mean = prefix + "_means_" + feature
    feature_sd = prefix + "_dev_" + feature
    df[prefix+"_sdom_"+feature] = df[feature] - df[feature_mean]
    df[prefix+"_sdom_"+feature] = df[prefix+"_sdom_"+feature].divide(df[feature_sd])
    df[prefix+"_sdom_"+feature] = df[prefix+"_sdom_"+feature].replace([np.inf, -np.inf], 0)
    if df[prefix+"_sdom_"+feature].isna().any():
        n_of_nas = df[prefix+"_sdom_"+feature][df[prefix+"_sdom_"+feature].isna()].index
        logger.info(
            f"Replacing nan with zero for {prefix}_sdom_{feature} , number of entries: {len(n_of_nas)}"
        )
        df[prefix+"_sdom_"+feature].fillna(0, inplace=True)
    return df[prefix+"_sdom_"+feature]

def generate_sdom(df, feature_sdom_levels, compute_features):
    for prefix in feature_sdom_levels.keys():
        for feature in feature_sdom_levels[prefix]:
            df[prefix+"_sdom_"+feature] = calculate_sdom(df, feature, prefix)
    return df


def get_top_level_data(slide_id, per_ml):
    df = per_ml[per_ml["slide_id"] == slide_id].copy()
    if len(df) == 0:
        logger.info(f"No top level data found for the slide: {slide_id}")
        return None
    tube_id = df["tube_id"].values[0]
    patient_id = df["patient_id"].values[0]
    dapi_count = df["dapi_count_ocular"].values[0]
    wbc_count = df["wbc_count"].values[0]
    conversion_factor = wbc_count / dapi_count
    clinical_classification = df["classification"].values[0]
    top_data = pd.DataFrame([{
        "slide_id": slide_id,
        "tube_id": tube_id,
        "patient_id": patient_id,
        "dapi_count": dapi_count,
        "wbc_count": wbc_count,
        "conversion_factor": conversion_factor,
        "clinical_classification": clinical_classification
    }])
    return top_data

def merge_all(uncurated_events,
              frame_stats,
              slide_stats,):
    rows = len(frame_stats)
    slide_stats = pd.concat(
            [slide_stats] * rows,
            ignore_index=True,
        )
    final_df = frame_stats.merge(
            slide_stats, left_index=True, right_index=True, how="inner"
        )
    final_df = pd.merge(uncurated_events, final_df, on=["frame_id"])

    return final_df


def process_slide(slide_id,
                     events,
                       per_ml,
                        ocular_params,):
    """
    This method accepts the slide id, curated events and per ml data,
    drops the duplicates and drops unused columns, and gets the frame and slide level
    morphopmetrics and merges them along with event level data then reads ocular files,
    namely, rc-final1.rds, rc-final2.rds, rc-final3.rds, rc-final4.rds, to get the set
    of interesting events along with manually curated interesting events and then set
    of not interesting events from uncurated_not_interesting evenst from files and,
    manual_not_interesting to produce the final dataframe for that slide.

    Args:
    slide_id: str
    events: pd.DataFrame
    per_ml: pd.DataFrame
    ocular_params: dict

    Returns:
    final_df: pd.DataFrame

    """
    # Slice events corresponding to the slide
    manual_interesting = events[(events['slide_id'] == slide_id) & (events['interesting'] == 1)].copy()
    manual_not_interesting = events[(events['slide_id'] == slide_id) & (events['interesting'] == 0)].copy()
    manual_interesting["unique_id"] = (
        manual_interesting["slide_id"].astype(str) + "_" +
        manual_interesting["frame_id"].astype(str) + "_" +
        manual_interesting["cell_id"].astype(str)
        )
    manual_not_interesting["unique_id"] = (
        manual_not_interesting["slide_id"].astype(str) + "_" +
        manual_not_interesting["frame_id"].astype(str) + "_" +
        manual_not_interesting["cell_id"].astype(str)
        )
    top_data = get_top_level_data(slide_id, per_ml)

    uncurated_events = get_data_from_files(slide_id, ocular_params)
    if len(uncurated_events) != 0:
        uncurated_events["slide_id"] = slide_id

    uncurated_events["unique_id"] = (
        uncurated_events["slide_id"].astype(str) + "_" +
        uncurated_events["frame_id"].astype(str) + "_" +
        uncurated_events["cellx"].astype(str) + "_" +
        uncurated_events["celly"].astype(str)
    )
    uncurated_events.sort_values(by=["cell_id"],
                                  ascending=True, inplace=True)
    uncurated_events.drop_duplicates(subset=['unique_id'],
                                      keep='first', inplace=True)
    uncurated_events["unique_id"] = (
        uncurated_events["slide_id"].astype(str) + "_" +
        uncurated_events["frame_id"].astype(str) + "_" +
        uncurated_events["cell_id"].astype(str)
    )
    uncurated_events.drop_duplicates(subset=['unique_id'],
                                      keep='first', inplace=True)
    
    frame_stats = get_frame_morphs(slide_id, ocular_params)

    slide_stats = get_slide_stats(slide_id, ocular_params)

    uncurated_events = merge_all(uncurated_events,
                         frame_stats,
                         slide_stats)
    if ocular_params['drop_columns']:
        columns_to_keep = uncurated_events.columns
        if 'haralick' in ocular_params['drop_morphs']:
            columns_to_keep = [col for col in columns_to_keep if '.h.' not in col]
        if 'theta' in ocular_params['drop_morphs']:
            columns_to_keep = [col for col in columns_to_keep if 'theta' not in col]
            
        if 'blurred' in ocular_params['drop_morphs']:
            for channel in channel_map.keys():
                columns_to_keep = [col for col in columns_to_keep if 'B'+channel_map[channel] not in col]

        if 'multi_channel' in ocular_params['drop_morphs']:
            drop_keys = [''.join(pair) for pair in permutations(channel_map.values(), 2)]
            for key in drop_keys:
                columns_to_keep = [col for col in columns_to_keep if key not in col]

        if 'mean_sd_q05' in ocular_params['drop_morphs']:
            columns_to_keep=[col for col in columns_to_keep if ('.mean' in col) or ('.sd' in col) or ('.q05' in col) ]
        
        if 'mean_q05' in ocular_params['drop_morphs']:
            columns_to_keep=[col for col in columns_to_keep if ('.mean' in col) or ('.q05' in col) ]
        
        if 'remove_slide_level' in ocular_params['drop_morphs']:
            columns_to_keep=[col for col in columns_to_keep if 'slide' not in col ]
        
        if 'remove_frame_level' in ocular_params ['drop_morphs']:
            columns_to_keep=[col for col in columns_to_keep if 'frame' not in col ]

        # Adding the frame and slide id since it was dropped during morphs drop
        columns_to_keep+=["frame_id", "slide_id"]
        compute_features = []
        for prefix in feature_sdom_levels.keys():
            compute_features += [prefix +"_sdom_"+item for item in feature_sdom_levels[prefix]]
        columns_to_keep += compute_features

        uncurated_events = generate_sdom(uncurated_events, feature_sdom_levels, compute_features)


        uncurated_events = uncurated_events[columns_to_keep]

        uncurated_interesting = uncurated_events[(
        uncurated_events['unique_id'].isin(manual_interesting['unique_id']) )].copy()

        uncurated_not_interesting = uncurated_events[(
        ~(uncurated_events['unique_id'].isin(manual_interesting['unique_id'])) )]

        manual_not_interesting = uncurated_not_interesting[uncurated_not_interesting['unique_id'].isin(
                                                            manual_not_interesting['unique_id'])]
        
        uncurated_not_interesting = uncurated_not_interesting[~(uncurated_not_interesting['unique_id'].isin(
                                                            manual_not_interesting['unique_id']))]

        if ocular_params['not_int_sample_size'] > 0:
            if len(manual_not_interesting) > ocular_params['not_int_sample_size']:
                manual_not_interesting=manual_not_interesting.sample(n=ocular_params['not_int_sample_size'],
                                                                 random_state=42)
        uncurated_interesting["interesting"] = 1
        uncurated_not_interesting["interesting"] = 0
        manual_not_interesting["interesting"] = 0

        final_df = pd.concat([uncurated_interesting, uncurated_not_interesting,
                              manual_not_interesting])
        return final_df
        

def get_slides_data(slide_ids: list,
                    events: pd.DataFrame,
                    per_ml: pd.DataFrame,
                    ocular_params: dict = ocular_params,
                    ):
    slide_data = []
    for slide_id in slide_ids:
        slide_data.append(process_slide(slide_id = slide_id,
                                    events = events,
                                    per_ml = per_ml,
                                    ocular_params=ocular_params,))
    slide_data = pd.concat(slide_data, ignore_index=True)
    return slide_data

def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR ,
    output_path: Path = INTERIM_DATA_DIR,
    processed_path: Path = PROCESSED_DATA_DIR,
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
    events.to_csv(output_path / "events.csv", index=False)
    per_ml.to_csv(output_path / "per_ml.csv", index=False)
    logger.info(f"Found {len(events)} events in {len(slides_cohort)} slides")

    # List out all the processed slides
    slides_cohort = list(set(per_ml["slide_id"]))
    # Split the slides into train, val and test
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
    data = {}
    with open(f'{output_path}/train_slides.json', 'w') as f:
        json.dump(train_slides, f)
        data["train"] = train_slides
    with open(f'{output_path}/val_slides.json', 'w') as f:
        json.dump(val_slides, f)
        data["val"] = val_slides
    with open(f'{output_path}/test_slides.json', 'w') as f:
        json.dump(test_slides, f)
        data["test"] = test_slides
    
    for key, value in data.items():
        logger.info(f"processing {key} slides")
        slides = get_slides_data(slide_ids = value,
                                events = events,
                                per_ml = per_ml)
        slides.to_csv(PROCESSED_DATA_DIR / f"{key}_data.csv", index=False)
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    main()
