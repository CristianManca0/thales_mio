"""Script per il confronto dell'interpretabilità locale SHAP (Originale vs Avversario) sui dati Raw."""

import argparse
import json
import logging
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# Aggiungiamo la root del progetto al path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confronto Local Explainability (SHAP) Originale vs Avversario.")
    parser.add_argument("--model-name", type=str, default="HBOS", help="Nome del modello (es. HBOS)")
    parser.add_argument("--sample-idx", type=int, default=0, help="Indice del pacchetto da analizzare (es. 0)")
    parser.add_argument("--results-json", type=str,
                        default="results_raw/without_scaler/blackbox_attack/evolutionstrategy/hbos_top10_shap_directions.json",
                        help="Percorso al JSON con i risultati dell'attacco")
    parser.add_argument("--ds-path", type=str, default="data/datasets/attack_dataset_raw.csv",
                        help="Percorso del dataset raw")
    parser.add_argument("--top-features", type=int, default=20, help="Numero di feature da mostrare nel grafico")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    # 1. Caricamento Dati e Modello
    logger.info(f"Caricamento dataset {args.ds_path}...")
    dataset = pd.read_csv(base_dir / args.ds_path, sep=";", low_memory=False)

    if "ip.opt.time_stamp" in dataset.columns:
        dataset = dataset.drop(columns=["ip.opt.time_stamp"])

    orig_sample = dataset.iloc[args.sample_idx].copy()

    model_path = base_dir / f"data/trained_models_raw/without_scaler/{args.model_name}.pkl"
    logger.info(f"Caricamento modello {args.model_name}...")
    detector = joblib.load(model_path)

    # 2. Caricamento Risultati e Ricostruzione SEMPLIFICATA
    results_path = base_dir / args.results_json
    if not results_path.exists():
        logger.error(f"File risultati non trovato: {results_path}")
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        attack_results = json.load(f)

    if str(args.sample_idx) not in attack_results:
        logger.error(f"Sample {args.sample_idx} non trovato nel file dei risultati!")
        sys.exit(1)

    best_params = attack_results[str(args.sample_idx)]["best_params"]

    # --- LA MAGIA: Ricostruzione pulita e diretta ---
    adv_sample = orig_sample.copy()
    for col, val in best_params.items():
        adv_sample[col] = val
    # ------------------------------------------------

    # Casting di sicurezza minimo (nel caso pandas si confonda con l'update)
    for col in adv_sample.index:
        if type(orig_sample[col]) != type(adv_sample[col]) and not pd.isna(adv_sample[col]):
            try:
                adv_sample[col] = type(orig_sample[col])(adv_sample[col])
            except Exception:
                pass

    score_orig = detector.decision_function(pd.DataFrame([orig_sample]))[0]
    score_adv = detector.decision_function(pd.DataFrame([adv_sample]))[0]
    logger.info(f"Punteggio Originale : {score_orig:.4f}")
    logger.info(f"Punteggio Avversario: {score_adv:.4f}")

    # 3. Calcolo SHAP
    logger.info("Calcolo dei valori SHAP in corso (potrebbe richiedere qualche minuto)...")

    # Background per KernelExplainer
    X_background = shap.sample(dataset, 100)

    def predict_fn(X_array):
        df = pd.DataFrame(X_array, columns=dataset.columns)
        return detector.decision_function(df)

    explainer = shap.KernelExplainer(predict_fn, X_background)

    shap_orig = explainer.shap_values(pd.DataFrame([orig_sample]))
    shap_adv = explainer.shap_values(pd.DataFrame([adv_sample]))

    vals_orig = shap_orig[0] if isinstance(shap_orig, list) else shap_orig[0]
    vals_adv = shap_adv[0] if isinstance(shap_adv, list) else shap_adv[0]

    # 4. Creazione oggetti Explanation per i plot nativi di SHAP
    logger.info("Generazione dei grafici waterfall nativi di SHAP...")

    # SHAP ha bisogno del "base_value" (il valore atteso/medio) per i plot waterfall
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[0]

    # Oggetto Explanation per il pacchetto originale
    exp_orig = shap.Explanation(
        values=vals_orig,
        base_values=base_val,
        data=orig_sample.values,
        feature_names=list(dataset.columns)
    )

    # Oggetto Explanation per il pacchetto avversario
    exp_adv = shap.Explanation(
        values=vals_adv,
        base_values=base_val,
        data=adv_sample.values,
        feature_names=list(dataset.columns)
    )

    # 5. Salvataggio delle due immagini separate (grafici standard SHAP)
    out_dir = base_dir / "results_raw" / "without_scaler" / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- PLOT 1: PACCHETTO ORIGINALE ---
    plt.figure()  # Inizializza una figura pulita
    shap.plots.waterfall(exp_orig, max_display=args.top_features, show=False)
    plt.title(f"Originale | Score: {score_orig:.4f}", pad=20, fontweight='bold')
    out_orig = out_dir / f"{args.model_name}_sample{args.sample_idx}_shap_1_ORIGINAL.png"
    plt.savefig(out_orig, dpi=300, bbox_inches='tight')
    plt.close()

    # --- PLOT 2: PACCHETTO AVVERSARIO ---
    plt.figure()  # Inizializza un'altra figura pulita
    shap.plots.waterfall(exp_adv, max_display=args.top_features, show=False)
    plt.title(f"Avversario | Score: {score_adv:.4f}", pad=20, fontweight='bold')
    out_adv = out_dir / f"{args.model_name}_sample{args.sample_idx}_shap_2_ADVERSARIAL.png"
    plt.savefig(out_adv, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Grafici waterfall salvati con successo in:\n  -> {out_orig}\n  -> {out_adv}")