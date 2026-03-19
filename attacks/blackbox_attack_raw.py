"""Black-box feature-space attack guided by SHAP for network traffic classifiers."""

import argparse
import json
import logging
from pathlib import Path
import random
import sys
from typing import Dict, Any, List

import joblib
import nevergrad as ng
from nevergrad.optimization.base import ConfiguredOptimizer, Optimizer
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from ml_models import Detector, RawDetector

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ATTACK_TYPE_MAP = {
    0: "flooding",  # PFCP Flooding
    1: "session_deletion",  # PFCP Deletion
    2: "session_modification",  # PFCP Modification
    5: "upf_pdn0_fault",  # UPF PDN-0 Fault
    6: "restoration_teid",  # PFCP Restoration-TEID
}

ATTACK_FEATURES = {
    "flooding": ["pfcp.msg_type"],
    "session_deletion": [
        "pfcp.msg_type",
    ],
    "session_modification": [
        "pfcp.msg_type",
        "pfcp.seid",
    ],
    "upf_pdn0_fault": [
        "pfcp.node_id_ipv4",
        "pfcp.pdr_id",
        "pfcp.f_teid_flags.ch",
        "pfcp.f_teid_flags.ch_id",
        "pfcp.f_teid_flags.v6",
    ],
    "restoration_teid": [
        "pfcp.f_teid.teid",
        "pfcp.pdr_id",
    ],
}

# ---------------------------------------------------------------------
# MAPPATURA FEATURE MODIFICABILI
# (Definiamo solo i tipi. I limiti Nevergrad saranno calcolati dinamicamente)
# ---------------------------------------------------------------------
FEAT_MAPPING: Dict[str, Dict[str, str]] = {
    # ------------- IP -------------
    "ip.ttl": {"type": "int"},
    "ip.id": {"type": "hex"},
    "ip.len": {"type": "int"},
    "ip.checksum": {"type": "hex"},
    "ip.dsfield.dscp": {"type": "int"},
    "ip.flags.df": {"type": "bool_str"},
    # ------------- UDP -------------
    "udp.checksum": {"type": "hex"},
    "udp.srcport": {"type": "int"},
    "udp.port": {"type": "int"},
    # ------------- PFCP Booleans -------------
    "pfcp.apply_action.buff": {"type": "bool_str"},
    "pfcp.apply_action.forw": {"type": "bool_str"},
    "pfcp.apply_action.nocp": {"type": "bool_str"},
    "pfcp.apply_action.drop": {"type": "bool_str"},
    "pfcp.f_teid_flags.ch": {"type": "bool_str"},
    "pfcp.f_teid_flags.ch_id": {"type": "bool_str"},
    "pfcp.f_teid_flags.v6": {"type": "bool_str"},
    "pfcp.s": {"type": "bool_str"},
    "pfcp.ue_ip_address_flag.sd": {"type": "bool_str"},
    # ------------- PFCP IPv4 -------------
    "pfcp.f_seid.ipv4": {"type": "ipv4"},
    "pfcp.f_teid.ipv4_addr": {"type": "ipv4"},
    "pfcp.node_id_ipv4": {"type": "ipv4"},
    "pfcp.outer_hdr_creation.ipv4": {"type": "ipv4"},
    "pfcp.ue_ip_addr_ipv4": {"type": "ipv4"},
    # ------------- PFCP Hex -------------
    "pfcp.f_teid.teid": {"type": "hex"},
    "pfcp.outer_hdr_creation.teid": {"type": "hex"},
    "pfcp.seid": {"type": "hex"},
    "pfcp.flags": {"type": "hex"},
    # ------------- PFCP Numeric / Categorical IDs -------------
    "pfcp.cause": {"type": "float_int"},
    "pfcp.dst_interface": {"type": "float_int"},
    "pfcp.source_interface": {"type": "float_int"},
    "pfcp.node_id_type": {"type": "float_int"},
    "pfcp.pdn_type": {"type": "float_int"},
    "pfcp.ie_type": {"type": "float_int"},
    "pfcp.msg_type": {"type": "float_int"},
    "pfcp.pdr_id": {"type": "float_int"},
    "pfcp.response_to": {"type": "float_int"},
    "pfcp.precedence": {"type": "float_int"},
    # ------------- PFCP Metrics / Counters -------------
    "pfcp.duration_measurement": {"type": "float_int"},
    "pfcp.seqno": {"type": "float_int"},
    "pfcp.volume_measurement.dlnop": {"type": "float_int"},
    "pfcp.volume_measurement.dlvol": {"type": "float_int"},
    "pfcp.volume_measurement.tonop": {"type": "float_int"},
    "pfcp.volume_measurement.tovol": {"type": "float_int"},
    "pfcp.flow_desc_len": {"type": "float_int"},
    "pfcp.ie_len": {"type": "float_int"},
    "pfcp.lenght": {"type": "float_int"},
    # ------------- PFCP Timestamps -------------
    "pfcp.recovery_time_stamp": {"type": "float_int"},
    "pfcp.time_of_first_packet": {"type": "float_int"},
    "pfcp.time_of_last_packet": {"type": "float_int"},
    # ------------- Ordinal Encoded Strings -------------
    "pfcp.flow_desc": {"type": "float_int"},
    "pfcp.network_instance": {"type": "float_int"},
    "pfcp.user_id_imei": {"type": "float_int"},
}


class BlackBoxAttack:
    """Black-box attack for network traffic classifiers guided by SHAP."""

    def __init__(self, optimizer_cls: ConfiguredOptimizer, dataset: pd.DataFrame, features_to_attack: List[str]) -> None:
        """
        Create a BlackBoxAttack instance.
        """
        self._optimizer_cls: ConfiguredOptimizer = optimizer_cls
        self._optimizer: Optimizer = None
        self._query_budget: int = None
        self._dataset = dataset
        self._features_to_attack = features_to_attack

        random.seed(42)

    def run(
        self,
        sample_idx: int,
        sample: pd.Series,
        attack_type: int,
        detector: Detector,
        results_path: Path | str,
        query_budget: int = 100,
    ) -> None:
        """Run the black-box attack on a given sample."""
        self._query_budget = query_budget

        logger.info(f"Starting attack on sample {sample_idx}...")

        # --------------------------------------
        # [Step 1] Compute baseline predictions
        # --------------------------------------
        orig_score = detector.decision_function(pd.DataFrame([sample]))[0]
        logger.info(f"Original score: {orig_score}")

        if hasattr(detector, "_detector"):
            logger.info(f"Detector threshold: {detector._detector.threshold_}")
            if orig_score < detector._detector.threshold_:
                logger.info("Sample is already classified as benign. Skipping attack.")
                return
        else:
            y_pred = detector.predict(pd.DataFrame([sample]))
            if y_pred == 0:
                logger.info("Sample is already classified as benign. Skipping attack.")
                return

        # --------------------------
        # [Step 2] Set up optimizer
        # --------------------------
        self._optimizer = self._init_optimizer(attack_type)

        # --------------------
        # [Step 3] Run attack
        # --------------------
        loss = None
        for idx in range(self._query_budget):
            x = self._optimizer.ask()
            x_adv = self._apply_modifications(sample, x.value)
            loss = self._compute_loss(x_adv, detector)

            # logger.info(f"Iteration {idx + 1}/{self._query_budget}: loss = {loss}")

            if hasattr(detector, "_detector"):
                if loss < detector._detector.threshold_:
                    logger.info(f"Sample evaded the detector after {idx + 1} queries.")
                    break
            else:
                y_pred = detector.predict(pd.DataFrame([x_adv]))
                if y_pred == 0:
                    logger.info(f"Sample evaded the detector after {idx + 1} queries.")
                    break

            self._optimizer.tell(x, loss)

        # ----------------------
        # [Step 4] Save results
        # ----------------------
        recommendation = self._optimizer.provide_recommendation()
        best_params = recommendation.value
        best_loss = recommendation.loss

        evaded = False
        if hasattr(detector, "_detector"):
            evaded = bool(best_loss < detector._detector.threshold_)
        else:
            evaded = bool(detector.predict(pd.DataFrame([self._apply_modifications(sample, best_params)]))[0] == 0)

        self._save_results(
            sample_idx,
            attack_type,
            best_params,
            best_loss,
            results_path,
            evaded
        )

    def _init_optimizer(self, attack_type: int) -> Optimizer:
        params = {}

        # Iteriamo SOLO sulle top features identificate da SHAP
        for feature in self._features_to_attack:
            mapping = FEAT_MAPPING[feature]
            if feature not in self._dataset.columns:
                continue

            # Skip features that should not be modified to preserve the attack intent
            if feature in ATTACK_FEATURES.get(ATTACK_TYPE_MAP.get(attack_type, ""), []):
                continue

            feat_type = mapping["type"]
            valid_data = self._dataset[feature].dropna()

            if feat_type == "bool_str":
                params[feature] = ng.p.Choice([0, 1])

            elif feat_type == "ipv4":
                unique_ips = valid_data.unique().tolist()
                if not unique_ips:
                    unique_ips = ["192.168.1.1"]  # Fallback
                params[feature] = ng.p.Choice(unique_ips)

            else:
                # Gestione numerica: calcola limiti massimi e minimi dal dataset
                min_val = valid_data.min() if not valid_data.empty else 0
                max_val = valid_data.max() if not valid_data.empty else 100

                if min_val == max_val:
                    # Nevergrad va in errore se min == max in uno Scalar. Forziamo come Choice.
                    params[feature] = ng.p.Choice([min_val])
                else:
                    if feat_type in ["int", "hex", "float_int"]:
                        params[feature] = ng.p.Scalar(lower=int(min_val), upper=int(max_val)).set_integer_casting()
                    elif feat_type == "float":
                        params[feature] = ng.p.Scalar(lower=float(min_val), upper=float(max_val))

        p_dict = ng.p.Dict(**params)
        p_dict.random_state = np.random.RandomState(42)
        return self._optimizer_cls(parametrization=p_dict, budget=self._query_budget)

    def _save_results(
        self,
        sample_idx: int,
        attack_type: int,
        best_params: dict,
        best_loss: float,
        results_path: Path | str,
        evaded: bool,
    ) -> None:
        if Path(results_path).exists():
            with Path(results_path).open("r") as f:
                results = json.load(f)
        else:
            results = {}

        results[str(sample_idx)] = {
            "attack_type": attack_type,
            "best_params": best_params,
            "best_loss": best_loss,
            "evaded": bool(evaded),
        }

        with Path(results_path).open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

    def _compute_loss(self, x_adv: pd.Series, detector: Detector) -> float:
        return detector.decision_function(pd.DataFrame([x_adv]))[0]

    def _apply_modifications(self, sample: pd.Series, params: Dict[str, Any]) -> pd.Series:
        adv_sample = sample.copy()

        for feature, value in params.items():
            feat_type = FEAT_MAPPING[feature]["type"]

            if feat_type == "hex":
                adv_sample[feature] = hex(int(value))
            elif feat_type == "float_int":
                adv_sample[feature] = float(value)
            elif feat_type in ["int", "bool_str"]:
                adv_sample[feature] = int(value)
            else:
                adv_sample[feature] = value

        return adv_sample


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Black-box attack on network traffic classifiers guided by SHAP."
    )
    argparser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="The name of the trained model to attack",
    )
    argparser.add_argument(
        "--ds-path",
        type=str,
        default=None,
        help="The path to the attack dataset file (CSV format)",
        required=True,
    )
    argparser.add_argument(
        "--optimizer",
        type=str,
        choices=["ES", "DE"],
        default="ES",
        help="The Nevergrad optimizer to use for the attack",
    )
    argparser.add_argument(
        "--raw",
        action="store_true",
        help="Attack raw models and save to results_raw",
    )
    argparser.add_argument(
        "--top-k", type=int, default=10, help="Number of top SHAP modifiable features to attack (default: 10)"
    )
    args = argparser.parse_args()

    dataset = pd.read_csv(args.ds_path, sep=";", low_memory=False)
    labels = dataset["ip.opt.time_stamp"].copy()
    dataset = dataset.drop(columns=["ip.opt.time_stamp"])

    # 1. Recupero dei file SHAP in base al flag --raw
    suffix = "raw" if args.raw else "normal"
    base_res_dir = "results_raw" if args.raw else "results"

    shap_path = Path(__file__).parent.parent / f"{base_res_dir}/with_scaler/explainability/shap_features_{args.model_name}_{suffix}.json"

    if not shap_path.exists():
        print(f"ERRORE: File SHAP non trovato in {shap_path}.")
        print("Esegui prima explainability.py per questo modello!")
        sys.exit(1)

    with open(shap_path, "r", encoding="utf-8") as f:
        shap_data = json.load(f)

    # 2. Selezione Top K feature modificabili
    ordered_shap_features = [item["feature"] for item in shap_data]
    modifiable_shap_features = [feat for feat in ordered_shap_features if feat in FEAT_MAPPING]
    top_k_features = modifiable_shap_features[:args.top_k]

    print(f"\n--- SHAP GUIDED BLACKBOX ATTACK ({args.optimizer}) ---")
    print(f"Modello: {args.model_name}")
    print(f"Top {args.top_k} feature modificabili estratte da SHAP:")
    for i, feat in enumerate(top_k_features, 1):
        print(f"  {i}. {feat}")
    print("-" * 45 + "\n")

    if not top_k_features:
        print("Nessuna feature modificabile trovata nei risultati SHAP. Esco.")
        sys.exit(1)

    if args.optimizer == "ES":
        optimizer_cls = ng.optimizers.EvolutionStrategy(
            recombination_ratio=0.9,
            popsize=20,
            only_offsprings=False,
            offsprings=20,
            ranker="simple",
        )
    else:
        optimizer_cls = ng.optimizers.DifferentialEvolution(
            popsize=20,
            crossover="twopoints",
            propagate_heritage=True,
        )

    # Passiamo il dataset e la lista top_k all'inizializzazione del BlackBox per i bounds dinamici
    bb = BlackBoxAttack(optimizer_cls, dataset, top_k_features)

    # Gestione path dinamici in base al flag --raw
    if args.raw:
        model_path = Path(__file__).parent.parent / f"data/trained_models_raw/with_scaler/{args.model_name}.pkl"
        results_path = (
                Path(__file__).parent.parent
                / f"results_raw/with_scaler/blackbox_attack/{optimizer_cls.__class__.__name__.lower()}"
                / f"{args.model_name.lower()}_top{args.top_k}.json"
        )
    else:
        model_path = Path(__file__).parent.parent / f"data/trained_models/with_scaler/{args.model_name}.pkl"
        results_path = (
                Path(__file__).parent.parent
                / f"results/with_scaler/blackbox_attack/{optimizer_cls.__class__.__name__.lower()}"
                / f"{args.model_name.lower()}_top{args.top_k}.json"
        )

    detector = joblib.load(model_path)

    if results_path.exists():
        with results_path.open("r") as f:
            results = json.load(f)
    else:
        results = {}
        results_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, row in dataset.iterrows():
        if str(idx) in results:
            logger.info(f"Sample {idx} already attacked. Skipping.")
            continue

        bb.run(
            int(idx),
            row,
            int(labels.iloc[idx]),
            detector,
            results_path,
            query_budget=100,
        )