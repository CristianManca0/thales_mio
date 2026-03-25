"""Script to generate a comparative table of evasion scores for all models."""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

# Configurazione del logger
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EvasionTableGenerator:
    """Class to aggregate black-box attack results into a comparative table."""

    def __init__(self, dir_normal: Path, dir_raw: Path) -> None:
        self.dir_normal = dir_normal
        self.dir_raw = dir_raw

        # Mappiamo le colonne della tabella (Colonna Principale, Sottocolonna)
        # alla combinazione corretta di (Cartella, Suffisso del file)
        self.configs = {
            ("Normal", "All Features"): (self.dir_normal, "_topall.json"),
            ("Normal", "Top 10 Features"): (self.dir_normal, "_top10.json"),
            ("Raw", "All Features"): (self.dir_raw, "_topall.json"),
            ("Raw", "Top 10 Features"): (self.dir_raw, "_top10.json"),
        }

    def _calculate_score(self, file_path: Path) -> float | None:
        """Read a JSON result file and calculate the evasion score."""
        if not file_path.exists():
            return None

        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON in {file_path}")
            return None

        total_attacks = len(data)
        if total_attacks == 0:
            return 0.0

        evaded_count = sum(1 for result in data.values() if result.get("evaded") is True)
        return (evaded_count / total_attacks) * 100

    def get_all_model_names(self) -> set:
        """Scan directories to find all unique model names by removing suffixes."""
        models = set()
        # Controlliamo la cartella normal
        if self.dir_normal.exists():
            for json_file in self.dir_normal.glob("*.json"):
                name = json_file.name
                if name.endswith("_topall.json"):
                    models.add(name.replace("_topall.json", ""))
                elif name.endswith("_top10.json"):
                    models.add(name.replace("_top10.json", ""))

        # Controlliamo la cartella raw per sicurezza
        if self.dir_raw.exists():
            for json_file in self.dir_raw.glob("*.json"):
                name = json_file.name
                if name.endswith("_topall.json"):
                    models.add(name.replace("_topall.json", ""))
                elif name.endswith("_top10.json"):
                    models.add(name.replace("_top10.json", ""))

        logger.info(f"Found {len(models)} unique models in {self.dir_normal}")
        logger.info(f"Found models: {models}")
        return models

    def generate_table(self) -> pd.DataFrame:
        """Generate the comparative table with MultiIndex columns."""
        models = sorted(list(self.get_all_model_names()))
        if not models:
            logger.warning(f"No JSON files found in the provided directories: {self.dir_normal} and {self.dir_raw}")
            return pd.DataFrame()

        rows = []
        for model in models:
            row_data = {("Model", ""): model}
            # Calcoliamo lo score per ogni configurazione usando la mappa
            for (main_col, sub_col), (directory, suffix) in self.configs.items():
                file_path = directory / f"{model}{suffix}"
                score = self._calculate_score(file_path)
                # Se il file esiste formattiamo il numero, altrimenti mettiamo N/A
                if score is not None:
                    row_data[(main_col, sub_col)] = round(score, 2)
                else:
                    row_data[(main_col, sub_col)] = pd.NA

            rows.append(row_data)

        # Creazione del DataFrame
        df = pd.DataFrame(rows)
        # Impostiamo le colonne come MultiIndex per avere la struttura con le sottocolonne
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        return df


# ==========================================
# SEZIONE DI ESECUZIONE DEL CODICE
# ==========================================
if __name__ == "__main__":

    BASE_DIR = Path(__file__).parent.parent

    # PERCORSI DELLE CARTELLE (Hardcodati in base alla tua struttura)
    DIR_NORMAL = BASE_DIR / "results" / "with_scaler" / "blackbox_attack" / "evolutionstrategy"
    DIR_RAW = BASE_DIR / "results_raw" / "without_scaler" / "blackbox_attack" / "evolutionstrategy"

    # Nome del file in cui verrà salvata la tabella finale
    OUTPUT_CSV_PATH = Path("evasion_scores_table.csv")
    logger.info("Starting table generation...")

    # Creiamo l'oggetto passando i percorsi
    generator = EvasionTableGenerator(
        dir_normal=DIR_NORMAL,
        dir_raw=DIR_RAW
    )

    # Generiamo la tabella
    df_table = generator.generate_table()

    if not df_table.empty:
        # Stampiamo la tabella in formato testo leggibile nel terminale
        logger.info("Tabella generata con successo:\n\n" + df_table.to_string(index=False, na_rep="N/A") + "\n")

        # Salviamo la tabella in CSV
        df_table.to_csv(OUTPUT_CSV_PATH, index=False)
        logger.info(f"Table successfully saved to {OUTPUT_CSV_PATH.absolute()}")
    else:
        logger.error("Could not generate table. Please check if the directory paths are correct.")