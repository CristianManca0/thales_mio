import logging
from pathlib import Path
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.base import BaseEstimator, RegressorMixin

sys.path.append(str(Path(__file__).parent.parent))
from ml_models import RawDetector

# Configurazione percorsi
DATA_PATH = Path(__file__).parent.parent / "data/datasets/test_dataset_raw.csv"
MODEL_PATH = Path(__file__).parent.parent / "data/trained_models_raw/with_scaler/HBOS.pkl"
FIGURES_DIR = Path(__file__).parent.parent / "results_raw/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

class PyODWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, detector):
        self.detector = detector

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.detector.decision_function(X)

    def __sklearn_is_fitted__(self):
        return True

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags

if __name__ == "__main__":
    # 1. Caricamento dati e modello
    df = pd.read_csv(DATA_PATH, sep=";", low_memory=False)
    detector = joblib.load(MODEL_PATH)

    # 2. Selezione del background (solo campioni malevoli come nel paper)
    # Nel tuo dataset, ip.opt.time_stamp non è NaN per gli attacchi
    X_malicious = df[~df["ip.opt.time_stamp"].isna()].drop(columns=["ip.opt.time_stamp"])
    X_background = X_malicious.sample(n=len(X_malicious), random_state=42)

    # Ordiniamo le colonne come si aspetta il modello (fit time)
    X_background = X_background[sorted(X_background.columns)]

    # 3. Selezione feature da analizzare (es. quelle critiche del paper)
    features_to_plot = ["pfcp.msg_type", "pfcp.cause", "pfcp.pdr_id"]

    print(f"Generazione PDP per: {features_to_plot}...")

    # 4. Creazione del plot
    # Nota: passiamo il wrapper di PyOD perché sklearn richiede un regressore
    wrapper = PyODWrapper(detector._detector)
    fig, ax = plt.subplots(figsize=(15, 5))
    display = PartialDependenceDisplay.from_estimator(
        wrapper,
        X_background,
        features=features_to_plot,
        kind="average",  # "average" è il PDP globale chiesto dal paper
        ax=ax
    )

    plt.suptitle("Partial Dependence Plots (Background: Malicious Samples)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_file = FIGURES_DIR / "pdp_analysis.pdf"
    plt.savefig(output_file)
    print(f"Grafico salvato in: {output_file}")
    plt.show()