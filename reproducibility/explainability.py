import argparse
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

from ml_models import Detector, RawDetector
from preprocessing import Preprocessor, RawPreprocessor

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

VAL_SIZE = 0.50
RANDOM_STATE = 42

def _convert_labels_to_binary(labels: pd.Series, expected_len: int) -> np.ndarray:
    if labels is not None:
        return (~pd.isna(labels)).astype(int).values
    else:
        return np.zeros(expected_len, dtype=int)


def explain_model(model_name: str, use_raw: bool):
    """
    Explain the top features of a model using SHAP.

    Parameters:
    - model_name: the name of the model to explain (e.g., 'HBOS', 'IForest').
    - use_raw: whether to evaluate the raw model/dataset or the normal one.
    """
    if use_raw:
        model_path = Path(__file__).parent.parent / f"data/trained_models_raw/with_scaler/{model_name}.pkl"
        test_path = Path(__file__).parent.parent / "data/datasets/test_dataset_raw.csv"
        results_dir = Path(__file__).parent.parent / "results_raw/explainability"
    else:
        model_path = Path(__file__).parent.parent / f"data/trained_models/with_scaler/{model_name}.pkl"
        test_path = Path(__file__).parent.parent / "data/datasets/test_dataset.csv"
        results_dir = Path(__file__).parent.parent / "results/explainability"

    if not model_path.exists():
        logger.error(f"Model not found at: {model_path}")
        return

    if not test_path.exists():
        logger.error(f"Test dataset not found at: {test_path}")
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    logger.info(f"Loading data from {test_path}...")
    df_test = pd.read_csv(test_path, sep=";", low_memory=False)

    # 2. Preprocess data as done during evaluation
    X_test = df_test.drop(columns=["ip.opt.time_stamp"], errors="ignore")
    y_test = _convert_labels_to_binary(df_test.get("ip.opt.time_stamp", None), len(df_test))

    _, X_ts, _, y_ts = train_test_split(
        X_test,
        y_test,
        test_size=1 - VAL_SIZE,
        stratify=y_test,
        random_state=RANDOM_STATE,
    )

    if use_raw:
        processor = RawPreprocessor()
    else:
        processor = Preprocessor()

    X_ts = processor.test(X_ts)

    # 3. Load model
    logger.info(f"Loading model {model_name} from {model_path}...")
    detector = joblib.load(model_path)

    # SHAP explainer
    # Sample background for SHAP to speed up computation
    X_background = X_ts.sample(n=min(100, len(X_ts)), random_state=RANDOM_STATE)
    X_explain = X_ts.sample(n=min(500, len(X_ts)), random_state=RANDOM_STATE)

    # Sort columns just to be sure it matches model expectations
    X_background = X_background[sorted(X_background.columns)]
    X_explain = X_explain[sorted(X_explain.columns)]

    # SHAP summary plot requires numeric values to properly map colors to feature values.
    # Convert category dtypes to numeric if any are present.
    cat_cols = X_explain.select_dtypes(include=["category"]).columns
    if not cat_cols.empty:
        logger.info(f"Converting category columns to numeric for SHAP plot: {list(cat_cols)}")
        for col in cat_cols:
            X_explain[col] = pd.to_numeric(X_explain[col], errors='coerce')

    logger.info(f"Computing SHAP values for {model_name} (use_raw={use_raw})...")

    # Most PyOD models decision_function works with numpy arrays
    def model_predict(data):
        return detector._detector.decision_function(data)

    try:
        explainer = shap.KernelExplainer(model_predict, X_background)
        shap_values = explainer.shap_values(X_explain)

        # Save summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_explain, show=False, max_display=10)

        suffix = "raw" if use_raw else "normal"
        output_file = results_dir / f"shap_summary_{model_name}_{suffix}.png"

        plt.title(f"SHAP Top 10 Features - {model_name} ({suffix})")
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()

        logger.info(f"Successfully saved SHAP summary plot to {output_file}")

    except Exception as e:
        logger.error(f"Error computing SHAP for {model_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain PyOD models using SHAP.")
    parser.add_argument("--model", type=str, default="HBOS", help="Name of the model to explain (e.g. HBOS)")
    parser.add_argument("--raw", action="store_true", help="Evaluate the raw dataset/model")
    parser.add_argument("--normal", action="store_true", help="Evaluate the normal dataset/model")
    parser.add_argument("--all", action="store_true", help="Evaluate both raw and normal")

    args = parser.parse_args()

    if args.all or args.raw:
        explain_model(args.model, use_raw=True)

    if args.all or args.normal:
        explain_model(args.model, use_raw=False)

    if not (args.all or args.raw or args.normal):
        # Default to both
        explain_model(args.model, use_raw=True)
        explain_model(args.model, use_raw=False)
