import argparse
import json
import joblib
import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))

from ml_models import Detector, RawDetector, EnsembleDetector
from preprocessing import Preprocessor, RawPreprocessor
from attacks.blackbox_attack_raw import _enforce_network_constraints, FEAT_MAPPING

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

VAL_SIZE = 0.50
RANDOM_STATE = 42

def _apply_modifications(sample: pd.Series, params: dict) -> pd.Series:
    adv_sample = sample.copy()

    for feature, value in params.items():
        if feature not in FEAT_MAPPING:
            continue

        feat_type = FEAT_MAPPING[feature]["type"]

        if feat_type == "hex":
            adv_sample[feature] = hex(int(value))
        elif feat_type == "float_int":
            adv_sample[feature] = float(value)
        elif feat_type in ["int", "bool_str"]:
            adv_sample[feature] = int(value)
        else:
            adv_sample[feature] = value

    adv_sample = _enforce_network_constraints(adv_sample, sample)
    return adv_sample

def _convert_labels_to_binary(labels: pd.Series, expected_len: int) -> np.ndarray:
    if labels is not None:
        return (~pd.isna(labels)).astype(int).values
    else:
        return np.zeros(expected_len, dtype=int)

def load_data(model_name: str, optimizer_name: str, top_k: int, use_raw: bool = True):
    if use_raw:
        models_dir = Path(__file__).parent.parent / "data/trained_models_raw/without_scaler"
        test_path = Path(__file__).parent.parent / "data/datasets/test_dataset_raw.csv"
        attack_path = Path(__file__).parent.parent / "data/datasets/attack_dataset_raw.csv"
        opt_dir = "evolutionstrategy" if optimizer_name == "ES" else "differentialevolution"
        results_dir = Path(__file__).parent.parent / f"results_raw/without_scaler/blackbox_attack/{opt_dir}"
        processor = RawPreprocessor()
    else:
        models_dir = Path(__file__).parent.parent / "data/trained_models/without_scaler"
        test_path = Path(__file__).parent.parent / "data/datasets/test_dataset.csv"
        attack_path = Path(__file__).parent.parent / "data/datasets/attack_dataset.csv"
        opt_dir = "evolutionstrategy" if optimizer_name == "ES" else "differentialevolution"
        results_dir = Path(__file__).parent.parent / f"results/without_scaler/blackbox_attack/{opt_dir}"
        processor = Preprocessor()

    model_path = models_dir / f"{model_name}.pkl"
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return None, None, None, None, None

    detector = joblib.load(model_path)

    df_test = pd.read_csv(test_path, sep=";", low_memory=False)
    df_attack = pd.read_csv(attack_path, sep=";", low_memory=False)

    results_file = results_dir / f"{model_name.lower()}_top{top_k}.json"
    if not results_file.exists():
         logger.error(f"Attack results file not found: {results_file}")
         return None, None, None, None, None

    with open(results_file, "r") as f:
        attack_results = json.load(f)

    return detector, df_test, df_attack, attack_results, processor

def compute_shap_comparison(model_name: str, detector, df_test: pd.DataFrame, df_attack: pd.DataFrame,
                            attack_results: dict, sample_idx: int, processor, suffix: str):

    if str(sample_idx) not in attack_results:
        logger.error(f"Sample index {sample_idx} not found in attack results.")
        return

    result_data = attack_results[str(sample_idx)]
    best_params = result_data.get("best_params", {})

    if sample_idx >= len(df_attack):
        logger.error(f"Sample index {sample_idx} is out of bounds for attack dataset.")
        return

    orig_sample = df_attack.iloc[sample_idx].drop(labels=["ip.opt.time_stamp"], errors="ignore")
    adv_sample = _apply_modifications(orig_sample, best_params)

    X_test = df_test.drop(columns=["ip.opt.time_stamp"], errors="ignore")
    y_test = _convert_labels_to_binary(df_test.get("ip.opt.time_stamp", None), len(df_test))

    _, X_ts, _, _ = train_test_split(
        X_test, y_test, test_size=1 - VAL_SIZE, stratify=y_test, random_state=RANDOM_STATE
    )

    X_ts = processor.test(X_ts)

    # Process samples
    orig_df = processor.test(pd.DataFrame([orig_sample]))
    adv_df = processor.test(pd.DataFrame([adv_sample]))

    X_background = X_ts.sample(n=min(100, len(X_ts)), random_state=RANDOM_STATE)

    X_background = X_background[sorted(X_background.columns)]
    orig_df = orig_df[sorted(orig_df.columns)]
    adv_df = adv_df[sorted(adv_df.columns)]

    cat_cols = orig_df.select_dtypes(include=["category", "object"]).columns
    if not cat_cols.empty:
        for col in cat_cols:
            orig_df[col] = pd.to_numeric(orig_df[col], errors='coerce')
            adv_df[col] = pd.to_numeric(adv_df[col], errors='coerce')
            X_background[col] = pd.to_numeric(X_background[col], errors='coerce')

    def model_predict(data_array):
        df = pd.DataFrame(data_array, columns=orig_df.columns)
        return detector.decision_function(df, skip_preprocess=True)

    logger.info(f"Computing SHAP values for sample {sample_idx}...")
    explainer = shap.KernelExplainer(model_predict, X_background)

    shap_values_orig = explainer.shap_values(orig_df)
    shap_values_adv = explainer.shap_values(adv_df)

    generate_comparison_plot(model_name, sample_idx, shap_values_orig, shap_values_adv, orig_df, adv_df, suffix)


def generate_comparison_plot(model_name: str, sample_idx: int, shap_values_orig: np.ndarray,
                             shap_values_adv: np.ndarray, orig_df: pd.DataFrame,
                             adv_df: pd.DataFrame, suffix: str):

    results_dir = Path(__file__).parent.parent / f"results_{suffix}/without_scaler/explainability"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / f"shap_comparison_{model_name}_sample{sample_idx}_{suffix}.png"

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    plt.sca(axes[0])
    shap.summary_plot(shap_values_orig, orig_df, show=False, max_display=10, plot_type="bar")
    axes[0].set_title(f"Original Sample SHAP Values")

    plt.sca(axes[1])
    shap.summary_plot(shap_values_adv, adv_df, show=False, max_display=10, plot_type="bar")
    axes[1].set_title(f"Adversarial Sample SHAP Values")

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

    logger.info(f"Successfully saved SHAP comparison plot to {output_file}")

def run_pipeline(args):
    model_name = args.model_name
    sample_idx = args.sample_idx
    optimizer = args.optimizer
    top_k = args.top_k

    logger.info(f"========== STARTING RAW PIPELINE ==========")
    detector, df_test, df_attack, attack_results, processor = load_data(
        model_name, optimizer, top_k, use_raw=True
    )

    if detector is not None:
        compute_shap_comparison(model_name, detector, df_test, df_attack,
                                attack_results, sample_idx, processor, suffix="raw")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare SHAP explainability on original and adversarial samples.")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model to explain.")
    parser.add_argument("--sample-idx", type=int, required=True, help="Index of the sample in the attack dataset.")
    parser.add_argument("--optimizer", type=str, default="ES", choices=["ES", "DE"], help="Optimizer used for the attack.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top SHAP modifiable features used in attack.")

    args = parser.parse_args()
    run_pipeline(args)
