import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Mappatura degli attacchi per i log
ATTACK_TYPE_MAP = {
    0.0: "flooding",
    1.0: "session_deletion",
    2.0: "session_modification",
    5.0: "upf_pdn0_fault",
    6.0: "restoration_teid",
}


def extract_malicious_samples(input_path: Path, output_path: Path):
    if not input_path.exists():
        logger.error(f"File non trovato: {input_path}")
        return

    logger.info(f"Caricamento di {input_path.name}...")
    df = pd.read_csv(input_path, sep=";", low_memory=False)

    if "ip.opt.time_stamp" not in df.columns:
        logger.error("Colonna 'ip.opt.time_stamp' non trovata nel dataset.")
        return

    # Il traffico benigno ha NaN, il traffico malevolo ha il codice dell'attacco
    # Quindi teniamo solo le righe dove la colonna non è nulla
    malicious_mask = df["ip.opt.time_stamp"].notna()
    df_attack = df[malicious_mask].copy()

    logger.info(f"Estratti {len(df_attack)} campioni malevoli su un totale di {len(df)}.")

    # Mostriamo un riepilogo degli attacchi estratti
    counts = df_attack["ip.opt.time_stamp"].value_counts()
    for code, count in counts.items():
        attack_name = ATTACK_TYPE_MAP.get(code, f"Sconosciuto ({code})")
        logger.info(f"  - {attack_name}: {count} campioni")

    # Salvataggio del nuovo dataset
    df_attack.to_csv(output_path, sep=";", index=False)
    logger.info(f"Dataset salvato con successo in: {output_path}\n")


if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent.parent / "data/datasets"

    # Path per la versione STANDARD (Cleaned)
    test_normal = DATA_DIR / "test_dataset.csv"
    attack_normal = DATA_DIR / "attack_dataset.csv"

    # Path per la versione RAW
    test_raw = DATA_DIR / "test_dataset_raw.csv"
    attack_raw = DATA_DIR / "attack_dataset_raw.csv"

    logger.info("--- Creazione Attack Dataset (Standard) ---")
    extract_malicious_samples(test_normal, attack_normal)

    logger.info("--- Creazione Attack Dataset (RAW) ---")
    extract_malicious_samples(test_raw, attack_raw)

    logger.info("Operazione completata! Ora puoi usare questi file negli script di attacco.")