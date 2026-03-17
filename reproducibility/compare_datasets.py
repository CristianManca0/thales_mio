import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Percorsi e Label
DATA_DIR = Path(__file__).parent.parent / "data/datasets"
RAW_PATH = DATA_DIR / "train_dataset_raw.csv"
CLEAN_PATH = DATA_DIR / "train_dataset.csv"
LABEL_COL = "ip.opt.time_stamp"


def compare_datasets():
    if not RAW_PATH.exists() or not CLEAN_PATH.exists():
        logger.error("Dataset non trovati. Verifica i percorsi.")
        return

    df_raw = pd.read_csv(RAW_PATH, sep=";", low_memory=False)
    df_clean = pd.read_csv(CLEAN_PATH, sep=";", low_memory=False)

    print("\n" + "=" * 60)
    print("      ANALISI COMPARATIVA: RAW vs CLEAN (Standard)")
    print("=" * 60)

    # 1. Dimensioni
    print(f"\n[1] DIMENSIONI")
    print(f"RAW Set   : {df_raw.shape[0]} righe, {df_raw.shape[1]} colonne")
    print(f"CLEAN Set : {df_clean.shape[0]} righe, {df_clean.shape[1]} colonne")
    print(f"Feature rimosse: {df_raw.shape[1] - df_clean.shape[1]}")

    # 2. Integrità Dati (Escludendo la Label)
    print(f"\n[2] VALORI MANCANTI (Escludendo la colonna {LABEL_COL})")
    nan_raw = df_raw.drop(columns=[LABEL_COL], errors="ignore").isna().sum().sum()
    nan_clean = df_clean.drop(columns=[LABEL_COL], errors="ignore").isna().sum().sum()
    print(f"NaN totali in RAW  : {nan_raw}")
    print(f"NaN totali in CLEAN: {nan_clean}")

    # 3. Controllo Coerenza Tipi (Type Consistency)
    print(f"\n[3] COERENZA TIPI DI DATO (Sulle feature comuni)")
    common_cols = [c for c in df_raw.columns if c in df_clean.columns and c != LABEL_COL]
    mismatches = []

    for col in common_cols:
        type_raw = str(df_raw[col].dtype)
        type_clean = str(df_clean[col].dtype)
        if type_raw != type_clean:
            mismatches.append((col, type_raw, type_clean))

    if mismatches:
        print(f"{'Colonna':<35} | {'Tipo RAW':<12} | {'Tipo CLEAN':<12}")
        print("-" * 65)
        for col, tr, tc in mismatches:
            print(f"{col:<35} | {tr:<12} | {tc:<12}")
    else:
        print("Tutte le feature comuni hanno lo stesso tipo di dato.")

    # 4. Confronto Statistico Intelligente
    print(f"\n[4] STATISTICHE DETTAGLIATE (Feature critiche)")
    # Selezioniamo alcune feature comuni per il test
    test_cols = [c for c in common_cols if "pfcp" in c][:5]

    for col in test_cols:
        print(f"\n>>> FEATURE: {col}")

        # Se una è numerica e l'altra è bool/object, mostriamo info separate per non fare casino
        is_num_raw = pd.api.types.is_numeric_dtype(df_raw[col])
        is_num_clean = pd.api.types.is_numeric_dtype(df_clean[col])

        if is_num_raw and is_num_clean:
            # Entrambe numeriche: usiamo describe classico
            stats = pd.concat([df_raw[col].describe(), df_clean[col].describe()], axis=1)
            stats.columns = ['RAW (Num)', 'CLEAN (Num)']
            print(stats.to_string())
        else:
            # Almeno una è categorica: mostriamo conteggi e valori unici
            raw_info = {
                "Unique": df_raw[col].nunique(),
                "Top": df_raw[col].mode()[0] if not df_raw[col].mode().empty else "N/A",
                "Dtype": str(df_raw[col].dtype)
            }
            clean_info = {
                "Unique": df_clean[col].nunique(),
                "Top": df_clean[col].mode()[0] if not df_clean[col].mode().empty else "N/A",
                "Dtype": str(df_clean[col].dtype)
            }
            res = pd.DataFrame({"RAW": raw_info, "CLEAN": clean_info})
            print(res)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    compare_datasets()