"""
Script to create RAW train and test datasets.
It performs only the fundamental machine learning steps without feature optimization:
1. Merges raw datasets to create train and test datasets.
2. (NO PROTOCOL FILTERING)
3. (NO USELESS OR CONSTANT COLUMNS REMOVAL)
4. Converts categoric columns to numeric.
5. Imputes missing values.
6. Restores categoric columns.
"""

import logging
from pathlib import Path
import sys

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.utils import (
    convert_to_numeric_raw,
    load_imputers_raw,
    restore_categoric_columns_raw, load_encoders_raw,
)

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    # Load initial raw datasets
    df1 = pd.read_csv(
        Path(__file__).parent.parent
        / "data/raw_datasets/dataset_1_cleaned.csv",
        sep=";",
        low_memory=False,
    )
    df2 = pd.read_csv(
        Path(__file__).parent.parent
        / "data/raw_datasets/dataset_2_cleaned.csv",
        sep=";",
        low_memory=False,
    )
    df3 = pd.read_csv(
        Path(__file__).parent.parent
        / "data/raw_datasets/dataset_3_cleaned.csv",
        sep=";",
        low_memory=False,
    )

    dataset_dir = Path(__file__).parent.parent / "data/datasets"
    dataset_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------
    # [Step 1] Merge datasets to create train and test datasets
    # ----------------------------------------------------------
    logger.info("Creating raw train and test datasets...")

    # Create train dataset concatenating dataset 1 and legitimate samples of dataset 2
    # legitimate samples of dataset 2 are those with label ip.opt.time_stamp = NaN,
    df_train = pd.concat(
        [df1, df2[df2["ip.opt.time_stamp"].isna()]], ignore_index=True
    )
    logger.info(f"Train dataset shape: {df_train.shape}")

    # Create test dataset with all samples of dataset 3 and attack samples of dataset 2
    df_test = pd.concat(
        [df2[~df2["ip.opt.time_stamp"].isna()], df3], ignore_index=True
    )
    logger.info(f"Test dataset shape: {df_test.shape}")

    # NO COLUMN FILTER: SKIPPED 'Filter out TCP and ICMP packets' STEP

    # --------------------------------------
    # [Step 2] Separate features and labels
    # --------------------------------------
    labels_train = df_train["ip.opt.time_stamp"].copy()
    labels_test = df_test["ip.opt.time_stamp"].copy()

    df_train.drop(columns=["ip.opt.time_stamp"], inplace=True)
    df_test.drop(columns=["ip.opt.time_stamp"], inplace=True)

    # NO COLUMN REMOVAL: SKIPPED 'drop_useless_columns' and 'drop_constant_columns' STEPS

    # ----------------------------------------------
    # [Step 3] Convert categoric columns to numeric
    # ----------------------------------------------
    logger.info("Converting categoric columns to numeric...")

    df_train_processed, cat_cols_train = convert_to_numeric_raw(df_train)
    df_test_processed, cat_cols_test = convert_to_numeric_raw(df_test)

    # -------------------------------
    # [Step 4] Impute missing values
    # -------------------------------
    logger.info("Imputing missing values and encoding strings...")

    simple_imputer, iter_imputer = load_imputers_raw(random_state=42)
    encoder = load_encoders_raw()  # Specular load

    # Identify valid columns for both sets
    valid_cols_train = df_train_processed.columns[df_train_processed.notna().any()]
    valid_cols_test = df_test_processed.columns[df_test_processed.notna().any()]
    common_valid_cols = valid_cols_train.intersection(valid_cols_test)

    # Separate features
    num_cols = df_train_processed[common_valid_cols].select_dtypes(include=["number"]).columns
    cat_cols = df_train_processed[common_valid_cols].select_dtypes(include=["category"]).columns
    # Identify string columns based on the tag from convert_to_numeric_raw
    string_cols = [col for col, tag in cat_cols_train if tag == "string" and col in common_valid_cols]

    # --- TRAIN DATA ---
    df_train[cat_cols] = simple_imputer.fit_transform(df_train_processed[cat_cols])
    df_train[num_cols] = iter_imputer.fit_transform(df_train_processed[num_cols])
    if string_cols:
        # Fit and transform strings specularly
        df_train[string_cols] = encoder.fit_transform(df_train[string_cols].astype(str))

    # --- TEST DATA ---
    df_test[cat_cols] = simple_imputer.transform(df_test_processed[cat_cols])
    df_test[num_cols] = iter_imputer.transform(df_test_processed[num_cols])
    if string_cols:
        # Transform strings specularly
        df_test[string_cols] = encoder.transform(df_test[string_cols].astype(str))

    # Save the encoder for the Preprocessor
    encoder_path = Path(__file__).parent.parent / "preprocessing/models_preprocessing/ordinal_encoder_raw.pkl"
    joblib.dump(encoder, encoder_path)

    # Round specific PFCP columns for consistency
    round_cols = [
        "pfcp.duration_measurement",
        "pfcp.ie_type",
        "pfcp.msg_type",
        "pfcp.pdr_id",
        "pfcp.response_to",
        "pfcp.volume_measurement.dlnop",
        "pfcp.volume_measurement.dlvol",
        "pfcp.volume_measurement.tonop",
        "pfcp.volume_measurement.tovol",
    ]
    for col in round_cols:
        if col in df_train.columns:
            df_train[col] = df_train[col].round().astype(float)
        if col in df_test.columns:
            df_test[col] = df_test[col].round().astype(float)

    # -----------------------------------
    # [Step 5] Restore categoric columns
    # -----------------------------------
    logger.info("Restoring categoric columns...")

    df_train = restore_categoric_columns_raw(
        df_train, cat_cols_train
    )
    df_test = restore_categoric_columns_raw(
        df_test, cat_cols_test
    )

    df_train["ip.opt.time_stamp"] = labels_train
    df_test["ip.opt.time_stamp"] = labels_test

    train_out_path = dataset_dir / "train_dataset_raw.csv"
    test_out_path = dataset_dir / "test_dataset_raw.csv"

    df_train.to_csv(train_out_path, sep=";", index=False)
    df_test.to_csv(test_out_path, sep=";", index=False)

    logger.info(f"Datasets successfully created and saved at:\n- {train_out_path}\n- {test_out_path}")