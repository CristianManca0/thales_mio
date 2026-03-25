"""
Script to create initial train and test datasets from raw datasets.
It performs the following steps:
1. Merges raw datasets to create train and test datasets.
2. Filters out TCP and ICMP packets.
3. Drops useless and constant columns.
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


def drop_tcp_and_icmp_packets(packets: pd.DataFrame) -> pd.DataFrame:
    # Filter out TCP (ip.proto == 6) and ICMP (ip.proto == 1) packets
    mask = packets["ip.proto"].isin([1, 6])
    filtered_packets = packets[~mask].copy()
    return filtered_packets


if __name__ == "__main__":
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
    logger.info("Creating train and test datasets...")

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

    # -----------------------------------------
    # [Step 2] Filter out TCP and ICMP packets
    # -----------------------------------------
    logger.info("Filtering out TCP and ICMP packets...")

    df_train_filtered = drop_tcp_and_icmp_packets(df_train)
    df_test_filtered = drop_tcp_and_icmp_packets(df_test)

    # --------------------------------------
    # [Step 3] Separate features and labels
    # --------------------------------------
    labels_train = df_train_filtered["ip.opt.time_stamp"].copy()
    labels_test = df_test_filtered["ip.opt.time_stamp"].copy()

    df_train_filtered.drop(columns=["ip.opt.time_stamp"], inplace=True)
    df_test_filtered.drop(columns=["ip.opt.time_stamp"], inplace=True)

    # ------------------------------
    # [Step 4] Drop useless columns -> SKIPPED
    # ------------------------------

    # ----------------------------------------------
    # [Step 5] Convert categoric columns to numeric
    # ----------------------------------------------
    logger.info("Converting categoric columns to numeric...")
    # ==========================================
    # BLOCCO DI DEBUG: Stampiamo i valori univoci
    # ==========================================
    for col in ["pfcp.end_time"]:
        if col in df_train_filtered.columns:
            train_vals = df_train_filtered[col].dropna().unique()
            logger.info(f"DEBUG {col} | TRAIN | Totale univoci: {len(train_vals)} | Primi 10: {train_vals[:10]}")
        if col in df_test_filtered.columns:
            test_vals = df_test_filtered[col].dropna().unique()
            logger.info(f"DEBUG {col} | TEST  | Totale univoci: {len(test_vals)} | Primi 10: {test_vals[:10]}")
    # ==========================================
    df_train_processed, cat_cols_train = convert_to_numeric_raw(df_train_filtered)
    # ==========================================
    # BLOCCO DI DEBUG: Che tipo gli ha assegnato?
    # ==========================================
    for col, dtype in cat_cols_train:
        if col in ["pfcp.end_time"]:
            logger.info(f"DEBUG TRAIN CATEGORY | Colonna: {col} | Tipo assegnato: {dtype}")
    # ==========================================
    df_test_processed, cat_cols_test = convert_to_numeric_raw(df_test_filtered)
    # ==========================================
    # BLOCCO DI DEBUG: Che tipo gli ha assegnato?
    # ==========================================
    for col, dtype in cat_cols_test:
        if col in ["pfcp.end_time"]:
            logger.info(f"DEBUG TEST CATEGORY | Colonna: {col} | Tipo assegnato: {dtype}")
    # ==========================================

    logger.info("Encoding strings...")
    encoder = load_encoders_raw()

    str_cols = [c for c, t in cat_cols_train if t == "string"]
    if str_cols:
        df_train_str = df_train_processed[str_cols].astype(str)
        df_test_str = df_test_processed[str_cols].astype(str)
        df_train_filtered[str_cols] = encoder.fit_transform(df_train_str)
        df_test_filtered[str_cols] = encoder.transform(df_test_str)
        encoder_path = Path(__file__).parent.parent / "preprocessing/models_preprocessing_raw/ordinal_encoder_raw.pkl"
        joblib.dump(encoder, encoder_path)

    # -------------------------------
    # [Step 6] Impute missing values
    # -------------------------------
    logger.info("Imputing missing values...")
    simple_imputer, iter_imputer = load_imputers_raw(random_state=42)
    encoder = load_encoders_raw()

    # Train data
    cat_cols = df_train_processed.select_dtypes(include=["category"]).columns
    num_cols = df_train_processed.select_dtypes(exclude=["category"]).columns

    # Fill completely empty columns with 0 to avoid imputation errors
    for df_p in [df_train_processed, df_test_processed]:
        all_nan_num = df_p[num_cols].columns[df_p[num_cols].isna().all()]
        if not all_nan_num.empty:
            logger.warning(f"Filling all-NaN numeric columns with 0: {list(all_nan_num)}")
            df_p[all_nan_num] = df_p[all_nan_num].fillna(0)
        all_nan_cat = df_p[cat_cols].columns[df_p[cat_cols].isna().all()]
        if not all_nan_cat.empty:
            logger.warning(f"Filling all-NaN categorical columns with 0: {list(all_nan_cat)}")
            df_p[all_nan_cat] = df_p[all_nan_cat].fillna(0)

    df_train_filtered[cat_cols] = simple_imputer.fit_transform(
        df_train_processed[cat_cols]
    )
    df_train_filtered[num_cols] = iter_imputer.fit_transform(
        df_train_processed[num_cols]
    )

    # Test data
    cat_cols = df_test_processed.select_dtypes(include=["category"]).columns
    num_cols = df_test_processed.select_dtypes(exclude=["category"]).columns

    df_test_filtered[cat_cols] = simple_imputer.transform(
        df_test_processed[cat_cols]
    )
    df_test_filtered[num_cols] = iter_imputer.transform(
        df_test_processed[num_cols]
    )

    round_cols = [
        # ONLY for RAW version
        "tcp.ack",
        "tcp.ack_raw",
        "tcp.analysis.bytes_in_flight",
        "tcp.analysis.push_bytes_sent",
        "tcp.completeness",
        "tcp.dstport",
        "tcp.hdr_len",
        "tcp.len",
        "tcp.nxtseq",
        "tcp.option_kind",
        "tcp.option_len",
        "tcp.options.timestamp.tsecr",
        "tcp.options.timestamp.tsval",
        "tcp.port",
        "tcp.seq",
        "tcp.seq_raw",
        "tcp.srcport",
        "tcp.stream",
        "tcp.window_size",
        "tcp.window_size_value",
        "udp.dstport",
        "udp.length",
        "udp.port",
        "udp.srcport",
        "udp.stream",

        "pfcp.end_time",
        "pfcp.recovery_time_stamp",
        "pfcp.time_of_first_packet",
        "pfcp.time_of_last_packet",
        "pfcp.cause",
        "pfcp.dst_interface",
        "pfcp.source_interface",
        "pfcp.flow_desc_len",
        "pfcp.ie_len",
        "pfcp.length",
        "pfcp.node_id_type",
        "pfcp.pdn_type",
        "pfcp.precedence",
        "pfcp.seqno",
        "pfcp.source_interface",
        "pfcp.user_id.imei",
        # COMMON
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
        if col in df_train_filtered.columns:
            df_train_filtered[col] = df_train_filtered[col].round().astype(float)
        if col in df_test_filtered.columns:
            df_test_filtered[col] = df_test_filtered[col].round().astype(float)

    # -----------------------------------
    # [Step 7] Restore categoric columns
    # -----------------------------------
    logger.info("Restoring categoric columns...")

    df_train_filtered = restore_categoric_columns_raw(
        df_train_filtered, cat_cols_train
    )
    df_test_filtered = restore_categoric_columns_raw(
        df_test_filtered, cat_cols_test
    )

    df_train_filtered["ip.opt.time_stamp"] = labels_train
    df_test_filtered["ip.opt.time_stamp"] = labels_test

    df_train_filtered.to_csv(
        dataset_dir / "train_dataset_raw.csv", sep=";", index=False
    )
    df_test_filtered.to_csv(
        dataset_dir / "test_dataset_raw.csv", sep=";", index=False
    )

    logger.info("Datasets created and saved successfully.")
