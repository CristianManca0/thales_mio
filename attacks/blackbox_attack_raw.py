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

from ml_models import RawDetector

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _enforce_network_constraints(sample: pd.Series, orig_sample: pd.Series) -> pd.Series:
    """
    Ricalcola le feature derivate per mantenere il pacchetto avversario
    fisicamente realistico e coerente con lo stack TCP/IP e PFCP.
    """
    adv_sample = sample.copy()
    old_sample = adv_sample.copy()

    # 1. Coerenza del QoS (ip.dsfield <- ip.dsfield.dscp)
    if "ip.dsfield.dscp" in adv_sample and "ip.dsfield" in adv_sample:
        orig_dsfield = int(str(orig_sample["ip.dsfield"]), 16) if pd.notnull(orig_sample["ip.dsfield"]) else 0
        orig_ecn = orig_dsfield & 0x03
        new_dscp = int(adv_sample["ip.dsfield.dscp"])
        adv_sample["ip.dsfield"] = hex((new_dscp << 2) | orig_ecn)

    # 2. Coerenza dei Flag IP (ip.flags <- ip.flags.df)
    if "ip.flags.df" in adv_sample and "ip.flags" in adv_sample:
        df_bit = int(adv_sample["ip.flags.df"])
        adv_sample["ip.flags"] = hex(0x02) if df_bit == 1 else hex(0x00)

    # 3. Coerenza Porte UDP (udp.port <- udp.srcport)
    if "udp.srcport" in adv_sample and "udp.port" in adv_sample:
        adv_sample["udp.port"] = adv_sample["udp.srcport"]

    # 5. Coerenza Volumetrie PFCP (Usage Reports)
    if "pfcp.volume_measurement.dlvol" in adv_sample and "pfcp.volume_measurement.tovol" in adv_sample:
        dlvol = float(adv_sample["pfcp.volume_measurement.dlvol"])
        tovol = float(adv_sample["pfcp.volume_measurement.tovol"])
        if dlvol > tovol:
            adv_sample["pfcp.volume_measurement.tovol"] = float(dlvol)

    if "pfcp.volume_measurement.dlnop" in adv_sample and "pfcp.volume_measurement.tonop" in adv_sample:
        dlnop = float(adv_sample["pfcp.volume_measurement.dlnop"])
        tonop = float(adv_sample["pfcp.volume_measurement.tonop"])
        if dlnop > tonop:
            adv_sample["pfcp.volume_measurement.tonop"] = float(dlnop)

    # 6. Coerenza Lunghezze (ip.len vs pfcp.lenght)
    if "pfcp.lenght" in adv_sample and "ip.len" in adv_sample:
        # Se c'è una discrepanza tra IP len e PFCP len, facciamo comandare PFCP
        # IP Len = 20(IPv4) + 8(UDP) + 4(PFCP base) + PFCP_Message_Length
        pfcp_len = float(adv_sample["pfcp.lenght"])
        adv_sample["ip.len"] = int(32 + pfcp_len)

    changes = {}
    for col in adv_sample.index:
        val_new = adv_sample[col]
        val_old = old_sample[col]
        if val_new != val_old:
            if pd.isna(val_new) and pd.isna(val_old):
                continue
            changes[col] = val_new

    '''if changes:
        print(f"\n[DEBUG CONSTRAINTS] Feature derivate ricalcolate per mantenere coerenza:")
        for k, v in changes.items():
            print(f"  -> {k}: {v}")'''

    return adv_sample


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

FEAT_MAPPING = {
    # ------------- IP -------------
    "ip.ttl": ng.p.Scalar(lower=2, upper=200).set_integer_casting(),
    "ip.id": ng.p.Scalar(lower=0, upper=65534).set_integer_casting(),
    "ip.len": ng.p.Scalar(lower=44, upper=653).set_integer_casting(),
    "ip.checksum": ng.p.Scalar(lower=54, upper=65525).set_integer_casting(),
    "ip.dsfield.dscp": ng.p.Scalar(lower=0, upper=63).set_integer_casting(),
    "ip.flags.df": ng.p.Choice([True, False]),

    # ------------- UDP -------------
    "udp.checksum": ng.p.Scalar(lower=2389, upper=41240).set_integer_casting(),
    "udp.srcport": ng.p.Scalar(lower=8805, upper=62434).set_integer_casting(),

    # ------------- PFCP Booleans -------------
    "pfcp.apply_action.buff": ng.p.Choice([True, False]),
    "pfcp.apply_action.forw": ng.p.Choice([True, False]),
    "pfcp.apply_action.nocp": ng.p.Choice([True, False]),
    "pfcp.apply_action.drop": ng.p.Choice([True, False]),
    "pfcp.f_teid_flags.ch": ng.p.Choice([True, False]),
    "pfcp.f_teid_flags.ch_id": ng.p.Choice([True, False]),
    "pfcp.f_teid_flags.v6": ng.p.Choice([True, False]),
    "pfcp.s": ng.p.Choice([True, False]),
    "pfcp.ue_ip_address_flag.sd": ng.p.Choice([True, False]),

    # ------------- PFCP IPv4 -------------
    "pfcp.f_seid.ipv4": ng.p.Choice(
        [
            "192.168.14.155",
            "192.168.130.144",
            "192.168.14.164",
            "192.168.14.153",
            "192.168.14.129",
            "192.168.14.150",
            "192.168.14.176",
            "192.168.130.176",
        ]
    ),
    "pfcp.f_teid.ipv4_addr": ng.p.Choice(
        [
            "192.168.130.144",
            "192.168.14.153",
            "192.168.14.150",
            "192.168.130.176",
            "192.168.14.162",
            "192.168.130.179",
        ]
    ),
    "pfcp.node_id_ipv4": ng.p.Choice(
        [
            "192.168.130.144",
            "192.168.14.164",
            "192.168.14.153",
            "192.168.14.129",
            "192.168.14.150",
            "192.168.14.176",
            "192.168.130.176",
        ]
    ),
    "pfcp.outer_hdr_creation.ipv4": ng.p.Choice(
        [
            "192.168.14.155",
            "192.168.130.178",
            "192.168.14.164",
            "192.168.130.138",
            "192.168.14.129",
            "192.168.130.139",
            "192.168.14.176",
            "192.168.130.186",
            "192.168.130.182",
            "192.168.130.179",
            "192.168.130.181",
        ]
    ),
    "pfcp.ue_ip_addr_ipv4": ng.p.Choice(
        [
            "10.45.0.4",
            "10.45.0.55",
            "10.45.0.63",
            "10.45.0.103",
            "10.45.5.12",
            "10.45.6.10",
            "10.45.1.90",
            "10.45.3.174",
            "10.45.4.226",
            "10.45.4.107",
            "10.45.4.97",
            "10.45.5.82",
            "10.45.4.70",
            "10.45.5.56",
            "10.45.4.100",
            "10.45.5.194",
            "10.45.3.38",
            "10.45.1.26",
            "10.45.3.182",
            "10.45.3.132",
            "10.45.4.135",
            "10.45.2.54",
            "10.45.3.36",
            "10.45.4.48",
            "10.45.4.93",
            "10.45.4.155",
            "10.45.3.214",
            "10.45.4.35",
            "10.45.3.0",
            "10.45.4.60",
            "10.45.4.223",
            "10.45.3.242",
            "10.45.5.67",
            "10.45.6.157",
        ]
    ),

    # ------------- PFCP Hex -------------
    "pfcp.f_teid.teid": ng.p.Scalar(
        lower=29, upper=65507
    ).set_integer_casting(),
    "pfcp.outer_hdr_creation.teid": ng.p.Scalar(
        lower=1, upper=6326
    ).set_integer_casting(),
    "pfcp.response_to": ng.p.Scalar(lower=1, upper=2834).set_integer_casting(),
    "pfcp.seid": ng.p.Scalar(lower=0, upper=4095).set_integer_casting(),
    "pfcp.flags": ng.p.Scalar(lower=32, upper=33).set_integer_casting(),

    # ------------- PFCP Numeric / Categorical IDs -------------
    "pfcp.cause": ng.p.Scalar(lower=1, upper=65).set_integer_casting(),
    "pfcp.dst_interface": ng.p.Scalar(lower=0, upper=1).set_integer_casting(),
    "pfcp.source_interface": ng.p.Scalar(lower=0, upper=1).set_integer_casting(),
    "pfcp.node_id_type": ng.p.Scalar(lower=0, upper=2).set_integer_casting(),
    "pfcp.pdn_type": ng.p.Scalar(lower=0, upper=1).set_integer_casting(),
    "pfcp.ie_type": ng.p.Scalar(lower=10, upper=96).set_integer_casting(),
    "pfcp.msg_type": ng.p.Scalar(lower=1, upper=57).set_integer_casting(),
    "pfcp.pdr_id": ng.p.Scalar(lower=1, upper=2).set_integer_casting(),
    "pfcp.precedence": ng.p.Scalar(lower=200, upper=255).set_integer_casting(),
    "pfcp.flow_desc": ng.p.Scalar(lower=-1, upper=2).set_integer_casting(),
    "pfcp.network_instance": ng.p.Scalar(lower=-1, upper=2).set_integer_casting(),
    "pfcp.user_id.imei": ng.p.Scalar(lower=4370816125816151.0, upper=4370816125816182.0).set_integer_casting(),

    # ------------- PFCP Metrics / Counters -------------
    "pfcp.duration_measurement": ng.p.Scalar(
        lower=1747212643, upper=1753894838
    ).set_integer_casting(),
    "pfcp.seqno": ng.p.Scalar(lower=0, upper=202364).set_integer_casting(),
    "pfcp.volume_measurement.dlnop": ng.p.Scalar(
        lower=0, upper=13195
    ).set_integer_casting(),
    "pfcp.volume_measurement.dlvol": ng.p.Scalar(
        lower=0, upper=17834134
    ).set_integer_casting(),
    "pfcp.volume_measurement.tonop": ng.p.Scalar(
        lower=0, upper=13195
    ).set_integer_casting(),
    "pfcp.volume_measurement.tovol": ng.p.Scalar(
        lower=0, upper=17834134
    ).set_integer_casting(),
    "pfcp.flow_desc_len": ng.p.Scalar(lower=34, upper=42).set_integer_casting(),
    "pfcp.ie_len": ng.p.Scalar(lower=1, upper=50).set_integer_casting(),
    "pfcp.length": ng.p.Scalar(lower=12, upper=621).set_integer_casting(),
    "pfcp.response_time": ng.p.Scalar(lower=2.0095e-05, upper=0.041239073),

    # ------------- PFCP Timestamps -------------
    #"pfcp.recovery_time_stamp": ng.p.Scalar(lower=1747207882, upper=1800000000).set_integer_casting(),
    #"pfcp.time_of_first_packet": ng.p.Scalar(lower=1747212464, upper=1800000000).set_integer_casting(),
    #"pfcp.end_time": ng.p.Scalar(lower=1747212642, upper=1800000000).set_integer_casting(),
    #"pfcp.time_of_last_packet": ng.p.Scalar(lower=1747212640, upper=1800000000).set_integer_casting(),
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
        detector: RawDetector,
        results_path: Path | str,
        shap_directions: dict = None,
        query_budget: int = 100,
    ) -> None:
        """Run the black-box attack on a given sample."""
        if shap_directions is None:
            shap_directions = {}

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
        self._optimizer = self._init_optimizer(attack_type, sample, shap_directions)

        # --------------------
        # [Step 3] Run attack
        # --------------------
        loss = None
        for idx in range(self._query_budget):
            x = self._optimizer.ask()
            x_adv = self._apply_modifications(sample, x.value)
            '''# --- INIZIO BLOCCO DEBUG ---
            if idx <= 3:
                print(f"\n[DEBUG] --- DUMP PACCHETTO AVVERSARIO ITERAZIONE 1 ---")
                anomalies = []
                for col in x_adv.index:
                    val = x_adv[col]
                    # Stampiamo solo le feature sotto attacco o quelle che sono palesemente rotte (NaN/None)
                    if col in self._features_to_attack or pd.isna(val) or val is None:
                        print(f"  -> {col}: {val} (Tipo: {type(val)})")
                        if pd.isna(val):
                            anomalies.append(col)
                print(f"[DEBUG] ------------------------------------------------\n")
            # --- FINE BLOCCO DEBUG ---'''

            loss = self._compute_loss(x_adv, detector)
            self._optimizer.tell(x, loss)

            logger.info(f"Iteration {idx + 1}/{self._query_budget}: loss = {loss}")
            if hasattr(detector, "_detector"):
                if loss < detector._detector.threshold_:
                    logger.info(f"Sample evaded the detector after {idx + 1} queries.")
                    break
            else:
                y_pred = detector.predict(pd.DataFrame([x_adv]))
                if y_pred == 0:
                    logger.info(f"Sample evaded the detector after {idx + 1} queries.")
                    break

        # ----------------------
        # [Step 4] Save results
        # ----------------------
        recommendation = self._optimizer.provide_recommendation()
        best_loss = recommendation.loss
        final_adv_sample = self._apply_modifications(sample, recommendation.value)
        final_saved_params = {}
        for col in final_adv_sample.index:
            val_adv = final_adv_sample[col]
            val_orig = sample[col]
            if pd.isna(val_adv) and pd.isna(val_orig):
                continue
            if val_adv != val_orig:
                if isinstance(val_adv, (np.integer, np.int64, np.int32)):
                    final_saved_params[col] = int(val_adv)
                elif isinstance(val_adv, (np.floating, np.float64, np.float32)):
                    final_saved_params[col] = float(val_adv)
                elif isinstance(val_adv, np.bool_):
                    final_saved_params[col] = bool(val_adv)
                else:
                    final_saved_params[col] = val_adv

        self._save_results(
            sample_idx,
            attack_type,
            final_saved_params,
            best_loss,
            results_path,
            bool(loss < detector._detector.threshold_) if hasattr(detector, "_detector") else bool(y_pred == 0),
        )

    def _init_optimizer(self, attack_type: int, sample: pd.Series, shap_directions: dict) -> Optimizer:
        """Inizializza l'ottimizzatore. Applica restrizioni dinamiche solo se shap_directions è fornito."""
        params = {}
        debug_header_printed = False

        for feature, parametrization in FEAT_MAPPING.items():
            # 1. Saltiamo le feature fondamentali per l'attacco
            if feature in ATTACK_FEATURES.get(ATTACK_TYPE_MAP.get(attack_type, ""), []):
                continue
            # 2. Filtro TOP-K
            if feature not in self._features_to_attack:
                continue
            if shap_directions and feature not in shap_directions:
                continue

            if isinstance(parametrization, ng.p.Scalar):
                # Estraiamo i limiti puliti (risolto il problema degli array di Numpy)
                lower_bound = float(np.ravel(parametrization.bounds[0])[0])
                upper_bound = float(np.ravel(parametrization.bounds[1])[0])
                val_str = str(sample[feature]).strip()
                if val_str.startswith("0x"):
                    # Se è un esadecimale, lo convertiamo prima in intero (base 16) e poi in float
                    valore_originale = float(int(val_str, 16))
                else:
                    # Altrimenti lo convertiamo normalmente
                    valore_originale = float(sample[feature])
                new_lower = lower_bound
                new_upper = upper_bound

                # 3. Logica Direzioni SHAP
                if shap_directions and feature in shap_directions:
                    direzione = shap_directions[feature]
                    if direzione == "decrease":
                        new_upper = min(upper_bound, valore_originale)
                    elif direzione == "increase":
                        new_lower = max(lower_bound, valore_originale)
                    if new_lower > new_upper:
                        new_lower, new_upper = new_upper, new_lower

                    # --- DEBUG COMPLETO SULLE DIREZIONI E RANGE ---
                    if not debug_header_printed:
                        logger.info(f"\n[SHAP BOUNDS DEBUG] Modifica dei range in corso:")
                        debug_header_printed = True

                    logger.info(f"  [{feature}] Valore pacchetto: {valore_originale} | SHAP: {direzione.upper()}")
                    logger.info(f"     Range standard : [{lower_bound}, {upper_bound}]")
                    if new_lower >= new_upper:
                        logger.info(
                            f"     Range ristretto: [{new_lower}, {new_upper}] -> COLLASSATO! Feature bloccata.")
                        continue
                    else:
                        logger.info(f"     Range ristretto: [{new_lower}, {new_upper}]")

                # Guardia di sicurezza finale per Nevergrad (salta se width <= 0)
                if new_lower >= new_upper:
                    continue

                new_param = ng.p.Scalar(lower=new_lower, upper=new_upper)
                if parametrization.integer and new_upper < 9e18:
                    new_param.set_integer_casting()
                params[feature] = new_param
            else:
                params[feature] = parametrization

        if debug_header_printed:
            logger.info("-" * 50 + "\n")

        params_dict = ng.p.Dict(**params)
        params_dict.random_state = np.random.RandomState(42)
        return self._optimizer_cls(
            parametrization=params_dict, budget=self._query_budget
        )

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

    def _compute_loss(self, x_adv: pd.Series, detector: RawDetector) -> float:
        return detector.decision_function(pd.DataFrame([x_adv]))[0]

    def _apply_modifications(self, sample: pd.Series, params: Dict[str, Any]) -> pd.Series:
        adv_sample = sample.copy()

        hex_features = [
            "ip.id", "ip.checksum", "udp.checksum",
            "pfcp.f_teid.teid", "pfcp.outer_hdr_creation.teid",
            "pfcp.seid", "pfcp.flags"
        ]

        for feature, value in params.items():
            if feature in hex_features:
                adv_sample[feature] = hex(max(0, int(float(value))))
            else:
                adv_sample[feature] = value

        adv_sample = _enforce_network_constraints(adv_sample, sample)
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
        default="data/datasets/attack_dataset_raw.csv",
        help="The path to the attack dataset file (CSV format)",
    )
    argparser.add_argument(
        "--optimizer",
        type=str,
        choices=["ES", "DE"],
        default="ES",
        help="The Nevergrad optimizer to use for the attack",
    )
    argparser.add_argument(
        "--top-k",
        default="all",
        help="Number of top SHAP modifiable features to attack (default: all)"
    )
    argparser.add_argument(
        "--use-shap-directions",
        action="store_true",
        help="Use SHAP directions to guide the attack (Grey-Box)",
    )
    args = argparser.parse_args()

    dataset = pd.read_csv(args.ds_path, sep=";", low_memory=False)
    labels = dataset["ip.opt.time_stamp"].copy()
    dataset = dataset.drop(columns=["ip.opt.time_stamp"])

    suffix = "raw"
    base_res_dir = "results_raw"
    shap_path = (Path(__file__).parent.parent
                 / f"{base_res_dir}/without_scaler/explainability/shap_features_{args.model_name}_{suffix}.json"
                 )
    if args.top_k != "all":
        if not shap_path.exists():
            logger.info(f"ERROR: SHAP Path not found: {shap_path}.")
            sys.exit(1)
        with open(shap_path, "r", encoding="utf-8") as f:
            shap_data = json.load(f)
        # Top K modifiable feature
        ordered_shap_features = [item["feature"] for item in shap_data]
        modifiable_shap_features = [feat for feat in ordered_shap_features if feat in FEAT_MAPPING]
        top_k_features = modifiable_shap_features[:int(args.top_k)]
        logger.info(f"\n--- SHAP GUIDED BLACKBOX ATTACK ({args.optimizer}) ---")
        logger.info(f"Top {args.top_k} feature modificabili estratte da SHAP:")
        for i, feat in enumerate(top_k_features, 1):
            logger.info(f"  {i}. {feat}")
        logger.info("-" * 45 + "\n")
    else:
        top_k_features = [feat for feat in FEAT_MAPPING]

    if not top_k_features:
        logger.info("No modifiable features found in SHAP data. Please check the SHAP results and FEAT_MAPPING.")
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

    model_path = Path(__file__).parent.parent / f"data/trained_models_raw/without_scaler/{args.model_name}.pkl"
    if args.use_shap_directions:
        results_path = (
                Path(__file__).parent.parent
                / f"results_raw/without_scaler/blackbox_attack/{optimizer_cls.__class__.__name__.lower()}"
                / f"{args.model_name.lower()}_top{args.top_k}_shap_directions.json"
        )
    else:
        results_path = (
                Path(__file__).parent.parent
                / f"results_raw/without_scaler/blackbox_attack/{optimizer_cls.__class__.__name__.lower()}"
                / f"{args.model_name.lower()}_top{args.top_k}.json"
        )

    detector = joblib.load(model_path)

    shap_directions = {}
    if args.use_shap_directions:
        shap_dir_path = Path(
            __file__).parent.parent / "results_raw" / "without_scaler" / "shap_directions" / f"{args.model_name}_shap_directions.json"
        if shap_dir_path.exists():
            with open(shap_dir_path, "r", encoding="utf-8") as f:
                shap_directions = json.load(f)
            logger.info(
                f"Loaded {len(shap_directions)} SHAP directions for {args.model_name}.")
        else:
            logger.warning(
                f"{shap_dir_path} does not exist. Proceeding without SHAP directions.")

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
            shap_directions=shap_directions,
            query_budget=100,
        )