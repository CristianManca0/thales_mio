import pandas as pd
from pathlib import Path
import json


def extract_features_info(dataset_path: Path) -> dict:
    """Estrae le informazioni sui range e sui tipi di dato da un dataset (basato sul tuo script originale)."""
    print(f"Caricamento dataset: {dataset_path}...")
    df = pd.read_csv(dataset_path, sep=";", low_memory=False)

    mapping = {}
    for col in df.columns:
        valid_data = df[col].dropna()
        if valid_data.empty:
            continue

        entry = {}
        unique_vals = valid_data.unique()
        n_unique = len(unique_vals)

        # 1. CONTROLLO STRINGHE
        if pd.api.types.is_string_dtype(valid_data):
            sample_val = str(valid_data.iloc[0]).strip()
            if sample_val.startswith("0x"):
                entry["type"] = "hex"
                try:
                    int_vals = valid_data.apply(
                        lambda x: int(str(x), 16) if str(x).startswith("0x") else int(float(x))
                    )
                    entry["min"] = int(int_vals.min())
                    entry["max"] = int(int_vals.max())
                except Exception as e:
                    entry["error"] = str(e)
            else:
                entry["type"] = "string"
                if n_unique <= 50:
                    entry["choices"] = unique_vals.tolist()

        # 2. CONTROLLO NUMERICI
        elif pd.api.types.is_numeric_dtype(valid_data):
            is_float = pd.api.types.is_float_dtype(valid_data)
            entry["type"] = "float" if is_float else "int"

            entry["min"] = float(valid_data.min()) if is_float else int(valid_data.min())
            entry["max"] = float(valid_data.max()) if is_float else int(valid_data.max())

            if n_unique <= 50:
                entry["choices"] = [float(x) if is_float else int(x) for x in unique_vals]

        mapping[col] = entry

    return mapping


def compare_datasets(path_normal: Path, path_raw: Path):
    """Confronta i range delle feature tra due dataset ed evidenzia le differenze."""
    print("--- FASE 1: ESTRAZIONE DATASET NORMALE ---")
    normal_info = extract_features_info(path_normal)

    print("\n--- FASE 2: ESTRAZIONE DATASET RAW ---")
    raw_info = extract_features_info(path_raw)

    print("\n--- FASE 3: CONFRONTO IN CORSO ---")

    # Troviamo quali colonne esistono in entrambi i file
    common_features = set(normal_info.keys()).intersection(set(raw_info.keys()))

    report = {
        "features_analizzate": len(common_features),
        "features_con_differenze": {},
        "features_identiche": []
    }

    for feat in common_features:
        norm_feat = normal_info[feat]
        raw_feat = raw_info[feat]

        diffs = {}

        # Confronto del minimo (se esiste in entrambi)
        if "min" in norm_feat and "min" in raw_feat:
            if norm_feat["min"] != raw_feat["min"]:
                diffs["min"] = {"normal": norm_feat["min"], "raw": raw_feat["min"]}

        # Confronto del massimo (se esiste in entrambi)
        if "max" in norm_feat and "max" in raw_feat:
            if norm_feat["max"] != raw_feat["max"]:
                diffs["max"] = {"normal": norm_feat["max"], "raw": raw_feat["max"]}

        # Confronto delle scelte categoriali (se esistono in entrambi)
        if "choices" in norm_feat and "choices" in raw_feat:
            # Ordiniamo le liste prima di confrontarle per evitare falsi positivi
            try:
                if sorted(norm_feat["choices"]) != sorted(raw_feat["choices"]):
                    diffs["choices"] = "Le liste di valori unici sono diverse tra i due dataset."
            except TypeError:
                # Se la lista contiene tipi misti e non può essere ordinata
                if set(norm_feat["choices"]) != set(raw_feat["choices"]):
                    diffs["choices"] = "Le liste di valori unici sono diverse tra i due dataset."

        if diffs:
            report["features_con_differenze"][feat] = diffs
        else:
            report["features_identiche"].append(feat)

    # Stampiamo un riassunto a schermo
    print(f"\n=== REPORT DI CONFRONTO FINALE ===")
    print(f"Feature in comune analizzate: {report['features_analizzate']}")
    print(f"Feature con range IDENTICI: {len(report['features_identiche'])}")
    print(f"Feature con DIFFERENZE: {len(report['features_con_differenze'])}")

    if report["features_con_differenze"]:
        print("\nAlcuni esempi di differenze trovate (max 5):")
        for i, (feat, diff) in enumerate(report["features_con_differenze"].items()):
            if i >= 5:
                print("  ...e altre (vedi il file JSON per i dettagli completi).")
                break
            print(f"  -> {feat}: {diff}")

    # Salviamo il report completo nella stessa cartella dello script
    out_file = Path(__file__).parent / "confronto_range.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"\nReport dettagliato salvato in: {out_file.absolute()}")


# ==========================================
# SEZIONE DI ESECUZIONE DEL CODICE
# ==========================================
if __name__ == "__main__":

    # Usiamo Path per trovare automaticamente la cartella principale del progetto
    BASE_DIR = Path(__file__).parent.parent

    # Inserisci qui i percorsi ai tuoi due file CSV (normale e raw).
    # Ho ipotizzato questi percorsi basandomi sui tuoi file precedenti.
    PATH_NORMAL = BASE_DIR / "data" / "datasets" / "test_dataset.csv"
    PATH_RAW = BASE_DIR / "data" / "datasets" / "test_dataset_raw.csv"

    # Avviamo il confronto!
    if not PATH_NORMAL.exists():
        print(f"Errore: File normale non trovato in {PATH_NORMAL}")
    elif not PATH_RAW.exists():
        print(f"Errore: File raw non trovato in {PATH_RAW}")
    else:
        compare_datasets(PATH_NORMAL, PATH_RAW)