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


def compute_shap_for_model(model_name: str, detector, X_background: pd.DataFrame, X_explain: pd.DataFrame,
                           results_dir: Path, suffix: str):
    """
    Computes and saves SHAP values, plots, and feature importances for a single model.
    """
    logger.info(f"Computing SHAP values for {model_name}...")

    # Wrapper per SHAP: SHAP passa un numpy array, ma i nostri detector richiedono un DataFrame
    def model_predict(data_array):
        df = pd.DataFrame(data_array, columns=X_explain.columns)
        return detector.decision_function(df, skip_preprocess=True)

    try:
        output_file = results_dir / f"shap_summary_{model_name}_{suffix}.png"
        # Skip execution if the file already exists to save time
        if output_file.exists():
            logger.info(
                f"SHAP summary plot already exists for {model_name} at {output_file}, skipping plot generation.")
        else:
            explainer = shap.KernelExplainer(model_predict, X_background)
            shap_values = explainer.shap_values(X_explain)

            # 1. Save summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_explain, show=False, max_display=10)

            plt.title(f"SHAP Top 10 Features - {model_name} ({suffix})")
            plt.tight_layout()
            plt.savefig(output_file, bbox_inches="tight")
            plt.close()

            logger.info(f"Successfully saved SHAP summary plot to {output_file}")

            # 2. Save feature importances to JSON
            mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
            feature_names = X_explain.columns.tolist()
            feature_importance = [
                {"feature": name, "importance": float(val)}
                for name, val in zip(feature_names, mean_abs_shap_values)
            ]

            # Sort by importance (descending)
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
            json_file = results_dir / f"shap_features_{model_name}_{suffix}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(feature_importance, f, indent=4)

            logger.info(f"Successfully saved SHAP features list to {json_file}\n")

    except Exception as e:
        logger.error(f"Error computing SHAP for {model_name}: {e}\n")


def run_pipeline(use_raw: bool, specific_model: str):
    """
    Loads data once and evaluates all models in the directory (or a specific one).
    """
    suffix = "raw" if use_raw else "normal"

    # Path configurati per la versione with_scaler
    if use_raw:
        models_dir = Path(__file__).parent.parent / "data/trained_models_raw/without_scaler"
        test_path = Path(__file__).parent.parent / "data/datasets/test_dataset_raw.csv"
        results_dir = Path(__file__).parent.parent / "results_raw/without_scaler/explainability"
        processor = RawPreprocessor()
    else:
        models_dir = Path(__file__).parent.parent / "data/trained_models/with_scaler"
        test_path = Path(__file__).parent.parent / "data/datasets/test_dataset.csv"
        results_dir = Path(__file__).parent.parent / "results/with_scaler/explainability"
        processor = Preprocessor()

    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return
    if not test_path.exists():
        logger.error(f"Test dataset not found at: {test_path}")
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    # Identifica i modelli da valutare
    if specific_model.lower() == "all":
        models_to_run = [p.stem for p in models_dir.glob("*.pkl")]
        # Exclude Ensemble detectors for now
        models_to_run = [m for m in models_to_run if not m.startswith("Ensemble")]
        # Exclude ABOD, FeatureBagging and INNE detector (too heavy for SHAP)
        models_to_run = [m for m in models_to_run if m != "ABOD" and
                                                     m != "FeatureBagging" and
                                                     m != "INNE" and
                                                     m != "LODA" and
                                                     m != "KNN"]
    else:
        models_to_run = [specific_model]

    if not models_to_run:
        logger.warning(f"No models found to process for '{suffix}' pipeline.")
        return

    # 1. Caricamento Dati (Eseguito una sola volta!)
    logger.info(f"========== STARTING {suffix.upper()} PIPELINE ==========")
    logger.info(f"Loading data from {test_path}...")
    df_test = pd.read_csv(test_path, sep=";", low_memory=False)

    X_test = df_test.drop(columns=["ip.opt.time_stamp"], errors="ignore")
    y_test = _convert_labels_to_binary(df_test.get("ip.opt.time_stamp", None), len(df_test))

    _, X_ts, _, y_ts = train_test_split(
        X_test, y_test, test_size=1 - VAL_SIZE, stratify=y_test, random_state=RANDOM_STATE
    )

    logger.info(f"Preprocessing data...")
    X_ts = processor.test(X_ts)

    # 2. Preparazione dei Sample per SHAP
    X_background = X_ts.sample(n=min(100, len(X_ts)), random_state=RANDOM_STATE)
    X_explain = X_ts.sample(n=min(500, len(X_ts)), random_state=RANDOM_STATE)

    X_background = X_background[sorted(X_background.columns)]
    X_explain = X_explain[sorted(X_explain.columns)]

    # Converte categorie in numeri (SHAP necessita di numeri per i plot)
    cat_cols = X_explain.select_dtypes(include=["category"]).columns
    if not cat_cols.empty:
        logger.info(f"Converting categorical columns to numeric for SHAP compatibility...")
        for col in cat_cols:
            X_explain[col] = pd.to_numeric(X_explain[col], errors='coerce')
            X_background[col] = pd.to_numeric(X_background[col], errors='coerce')

    # 3. Iterazione su tutti i modelli
    logger.info(f"Found {len(models_to_run)} models to evaluate.")

    for model_name in models_to_run:
        model_path = models_dir / f"{model_name}.pkl"
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            continue

        logger.info(f"---> Loading model {model_name} <---")
        detector = joblib.load(model_path)

        compute_shap_for_model(model_name, detector, X_background, X_explain, results_dir, suffix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain PyOD and Ensemble models using SHAP.")
    parser.add_argument("--model", type=str, default="all",
                        help="Name of the model to explain (e.g. HBOS) or 'all' to evaluate all models in the folder.")
    parser.add_argument("--raw", action="store_true", help="Evaluate the raw dataset/model pipeline.")
    parser.add_argument("--normal", action="store_true", help="Evaluate the normal dataset/model pipeline.")
    parser.add_argument("--all", action="store_true", help="Evaluate both raw and normal pipelines.")

    args = parser.parse_args()

    # Se l'utente non specifica nessun flag (--raw, --normal, --all), eseguiamo entrambi di default
    run_both = not (args.raw or args.normal or args.all)

    if args.all or args.raw or run_both:
        run_pipeline(use_raw=True, specific_model=args.model)

    if args.all or args.normal or run_both:
        run_pipeline(use_raw=False, specific_model=args.model)