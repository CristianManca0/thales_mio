import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))

from ml_models import RawDetector
from preprocessing import RawPreprocessor
from attacks.blackbox_attack_raw import _enforce_network_constraints

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

VAL_SIZE = 0.50
RANDOM_STATE = 42

def _apply_modifications(sample: pd.Series, params: Dict[str, Any]) -> pd.Series:
    adv_sample = sample.copy()

    hex_features = [
        "ip.id", "ip.checksum", "udp.checksum",
        "pfcp.f_teid.teid", "pfcp.outer_hdr_creation.teid",
        "pfcp.seid", "pfcp.flags"
    ]

    for feature, value in params.items():
        if feature in hex_features:
            adv_sample[feature] = hex(max(0, int(float(value))))
        else:
            adv_sample[feature] = value

    adv_sample = _enforce_network_constraints(adv_sample, sample)
    return adv_sample

def _convert_labels_to_binary(labels: pd.Series, expected_len: int) -> np.ndarray:
    if labels is not None:
        return (~pd.isna(labels)).astype(int).values
    else:
        return np.zeros(expected_len, dtype=int)

def main():
    parser = argparse.ArgumentParser(description="Compare SHAP explainability for an original vs adversarial sample.")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model (e.g. HBOS)")
    parser.add_argument("--sample-idx", type=int, required=True, help="Index of the sample in the attack dataset")
    parser.add_argument("--test-ds-path", type=str, required=True, help="Path to the test dataset CSV")
    parser.add_argument("--attack-ds-path", type=str, required=True, help="Path to the attack dataset CSV")
    parser.add_argument("--attack-results-path", type=str, required=True, help="Path to the attack results JSON")
    parser.add_argument("--output-plot", type=str, default="shap_comparison.png", help="Path to save the comparison plot")

    args = parser.parse_args()

    # Load Model
    model_path = Path(__file__).parent.parent / f"data/trained_models_raw/without_scaler/{args.model_name}.pkl"
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    detector = joblib.load(model_path)

    # Load Test Dataset for SHAP background
    logger.info(f"Loading test dataset from {args.test_ds_path}...")
    df_test = pd.read_csv(args.test_ds_path, sep=";", low_memory=False)
    X_test = df_test.drop(columns=["ip.opt.time_stamp"], errors="ignore")
    y_test = _convert_labels_to_binary(df_test.get("ip.opt.time_stamp", None), len(df_test))

    _, X_ts, _, y_ts = train_test_split(
        X_test, y_test, test_size=1 - VAL_SIZE, stratify=y_test, random_state=RANDOM_STATE
    )

    processor = RawPreprocessor()
    # We might have missing values before preprocessing if no fit was done properly on Dummy, but in real case it's fine.
    X_ts = processor.test(X_ts).fillna(0)
    X_background = X_ts.sample(n=min(100, len(X_ts)), random_state=RANDOM_STATE)
    X_background = X_background[sorted(X_background.columns)]

    cat_cols = X_background.select_dtypes(include=["category"]).columns
    if not cat_cols.empty:
        for col in cat_cols:
            X_background[col] = pd.to_numeric(X_background[col], errors='coerce')

    # Load Original Sample
    logger.info(f"Loading attack dataset from {args.attack_ds_path}...")
    df_attack = pd.read_csv(args.attack_ds_path, sep=";", low_memory=False)
    if args.sample_idx not in df_attack.index:
        logger.error(f"Sample index {args.sample_idx} not found in attack dataset")
        sys.exit(1)

    orig_sample_raw = df_attack.loc[args.sample_idx].drop(labels=["ip.opt.time_stamp"], errors="ignore")

    # Load Attack Results
    logger.info(f"Loading attack results from {args.attack_results_path}...")
    with open(args.attack_results_path, "r", encoding="utf-8") as f:
        attack_results = json.load(f)

    idx_str = str(args.sample_idx)
    if idx_str not in attack_results:
        logger.error(f"Sample index {args.sample_idx} not found in attack results")
        sys.exit(1)

    best_params = attack_results[idx_str]["best_params"]

    # Reconstruct Adversarial Sample
    adv_sample_raw = _apply_modifications(orig_sample_raw, best_params)

    # Preprocess Samples
    df_samples = pd.DataFrame([orig_sample_raw, adv_sample_raw])
    X_samples = processor.test(df_samples).fillna(0)
    X_samples = X_samples[sorted(X_samples.columns)]

    cat_cols = X_samples.select_dtypes(include=["category"]).columns
    if not cat_cols.empty:
        for col in cat_cols:
            X_samples[col] = pd.to_numeric(X_samples[col], errors='coerce')

    # Compute SHAP
    def model_predict(data_array):
        df = pd.DataFrame(data_array, columns=X_samples.columns)
        return detector.decision_function(df, skip_preprocess=True)

    explainer = shap.KernelExplainer(model_predict, X_background)
    logger.info("Computing SHAP values...")
    shap_values = explainer.shap_values(X_samples)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_vals = shap_values

    shap_orig = shap_vals[0]
    shap_adv = shap_vals[1]

    # Plot Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    num_features = min(10, len(X_samples.columns))

    # Get top features for original sample
    mean_abs_orig = np.abs(shap_orig)
    top_indices_orig = np.argsort(mean_abs_orig)[-num_features:]
    top_features_orig = X_samples.columns[top_indices_orig]
    top_shap_orig = shap_orig[top_indices_orig]

    axes[0].barh(range(num_features), top_shap_orig, color=['red' if x > 0 else 'blue' for x in top_shap_orig])
    axes[0].set_yticks(range(num_features))
    axes[0].set_yticklabels(top_features_orig)
    axes[0].set_title(f"Original Sample - Top {num_features} Features (Score: {model_predict(X_samples.iloc[[0]])[0]:.4f})")
    axes[0].set_xlabel("SHAP Value")

    # Get top features for adversarial sample
    mean_abs_adv = np.abs(shap_adv)
    top_indices_adv = np.argsort(mean_abs_adv)[-num_features:]
    top_features_adv = X_samples.columns[top_indices_adv]
    top_shap_adv = shap_adv[top_indices_adv]

    axes[1].barh(range(num_features), top_shap_adv, color=['red' if x > 0 else 'blue' for x in top_shap_adv])
    axes[1].set_yticks(range(num_features))
    axes[1].set_yticklabels(top_features_adv)
    axes[1].set_title(f"Adversarial Sample - Top {num_features} Features (Score: {model_predict(X_samples.iloc[[1]])[0]:.4f})")
    axes[1].set_xlabel("SHAP Value")

    plt.tight_layout()
    plt.savefig(args.output_plot, bbox_inches="tight")
    logger.info(f"Comparison plot saved to {args.output_plot}")

if __name__ == "__main__":
    main()
