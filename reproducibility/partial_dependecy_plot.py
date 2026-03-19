import logging
from pathlib import Path
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
from sklearn.base import BaseEstimator, RegressorMixin

sys.path.append(str(Path(__file__).parent.parent))

from preprocessing import RawPreprocessor

# Configurazione percorsi
DATA_PATH = Path(__file__).parent.parent / "data/datasets/test_dataset_raw.csv"
MODEL_PATH = Path(__file__).parent.parent / "data/trained_models_raw/with_scaler/HBOS.pkl"
FIGURES_DIR = Path(__file__).parent.parent / "results_raw/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

class PyODWrapper(BaseEstimator, RegressorMixin):
    """Wrapper to make PyOD models compatible with sklearn's partial_dependence."""
    def __init__(self, pyod_model, columns):
        self.pyod_model = pyod_model
        self.columns = columns
        self.is_fitted_ = True

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        df_X = pd.DataFrame(X, columns=self.columns)
        return self.pyod_model.decision_function(df_X.values)

if __name__ == "__main__":
    # 1. Caricamento dati e modello
    df = pd.read_csv(DATA_PATH, sep=";", low_memory=False)
    detector = joblib.load(MODEL_PATH)

    # 2. Selezione del background (solo campioni malevoli come nel paper)
    # Nel tuo dataset, ip.opt.time_stamp non è NaN per gli attacchi
    X_malicious = df[~df["ip.opt.time_stamp"].isna()].drop(columns=["ip.opt.time_stamp"], errors="ignore")
    X_background = X_malicious.sample(n=min(500, len(X_malicious)), random_state=42)

    # Applica il pre-processing come durante l'addestramento e valutazione
    processor = RawPreprocessor()
    X_background = processor.test(X_background)

    # Ordiniamo le colonne come si aspetta il modello (fit time)
    X_background = X_background[sorted(X_background.columns)]

    # 3. Selezione feature da analizzare (es. quelle critiche del paper)
    features_to_plot = ["pfcp.msg_type", "pfcp.cause", "pfcp.pdr_id"]

    # Filter only the features that actually exist in the processed dataset
    available_features = X_background.columns.tolist()
    features_to_plot = [f for f in features_to_plot if f in available_features]

    print(f"Generazione PDP per: {features_to_plot}...")

    # 4. Creazione del plot calcolando la dipendenza parziale manualmente
    wrapped_model = PyODWrapper(detector._detector, X_background.columns)

    fig, axes = plt.subplots(1, len(features_to_plot), figsize=(5 * len(features_to_plot), 5))
    if len(features_to_plot) == 1:
        axes = [axes]

    for i, feature in enumerate(features_to_plot):
        print(f"Calcolo partial dependence per {feature}...")

        # Calculate PDP manually using scikit-learn
        pdp_results = partial_dependence(
            wrapped_model,
            X_background,
            [feature],
            grid_resolution=50,
            kind="average"
        )

        # Plot the calculated PDP
        ax = axes[i]

        # Extract the values from the dictionary returned by partial_dependence
        # Scikit-learn >= 1.2 returns 'average' and 'values'
        values = pdp_results['values'][0]
        average = pdp_results['average'][0]

        ax.plot(values, average, marker='o', linestyle='-')
        ax.set_xlabel(feature)
        ax.set_ylabel("Partial dependence (Anomaly Score)")
        ax.set_title(f"PDP: {feature}")
        ax.grid(True, linestyle="--", alpha=0.7)

    plt.suptitle("Partial Dependence Plots (Background: Malicious Samples)")
    plt.tight_layout()

    output_file = FIGURES_DIR / "pdp_analysis.pdf"
    plt.savefig(output_file)
    print(f"Grafico salvato in: {output_file}")
    plt.show()