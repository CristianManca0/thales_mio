import pandas as pd
from pathlib import Path
import json


def extract_all_features(dataset_path: Path):
    print(f"Caricamento dataset: {dataset_path}...")
    df = pd.read_csv(dataset_path, sep=";", low_memory=False)

    final_mapping = {}

    print(f"Trovate {len(df.columns)} colonne. Estrazione a tappeto in corso...\n")

    for col in df.columns:
        valid_data = df[col].dropna()
        if valid_data.empty:
            continue

        entry = {}
        unique_vals = valid_data.unique()
        n_unique = len(unique_vals)

        # 1. CONTROLLO STRINGHE (Hex, IPv4, Timestamp, Categoriali)
        if pd.api.types.is_string_dtype(valid_data):
            sample_val = str(valid_data.iloc[0]).strip()

            # È un esadecimale? (es. 0x20)
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

            # È un IPv4? (es. 192.168.1.1)
            elif sample_val.count('.') == 3 and sample_val.replace('.', '').isdigit():
                entry["type"] = "ipv4"
                entry["choices"] = unique_vals.tolist()

            # È un Timestamp? (es. 2024-05-18 10:30:15)
            elif "-" in sample_val and ":" in sample_val and len(sample_val) > 10:
                entry["type"] = "timestamp"
                try:
                    dt_series = pd.to_datetime(valid_data, errors='coerce').dropna()
                    entry["min"] = float(dt_series.min().timestamp())
                    entry["max"] = float(dt_series.max().timestamp())
                except Exception as e:
                    entry["error"] = str(e)

            # Altrimenti è una stringa categoriale generica
            else:
                entry["type"] = "string"
                if n_unique <= 50:
                    entry["choices"] = unique_vals.tolist()

        # 2. CONTROLLO NUMERICI (Int, Float, Booleani)
        elif pd.api.types.is_numeric_dtype(valid_data):
            is_float = pd.api.types.is_float_dtype(valid_data)
            entry["type"] = "float" if is_float else "int"

            # Estrazione Min e Max
            entry["min"] = float(valid_data.min()) if is_float else int(valid_data.min())
            entry["max"] = float(valid_data.max()) if is_float else int(valid_data.max())

            # Se ci sono pochi valori unici (es. Booleani, Flag, Cause), li salviamo tutti
            if n_unique <= 50:
                entry["choices"] = [float(x) if is_float else int(x) for x in unique_vals]

        final_mapping[col] = entry

    # Salvataggio su file JSON
    output_file = Path(__file__).parent / "dataset_full_anatomy.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_mapping, f, indent=4)

    print(f"Operazione completata! Analizzate {len(final_mapping)} feature valide.")
    print(f"Risultato salvato in: {output_file}")


if __name__ == "__main__":
    DS_PATH = Path(__file__).parent.parent / "data/datasets/test_dataset_raw.csv"
    extract_all_features(DS_PATH)