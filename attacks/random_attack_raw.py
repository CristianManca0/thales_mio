"""Simple random feature-space attack guided by SHAP explainability."""

import argparse
import ipaddress
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List

import pandas as pd
import joblib

sys.path.append(str(Path(__file__).parent.parent))

from ml_models import Detector, RawDetector


def _enforce_network_constraints(sample: pd.Series, orig_sample: pd.Series) -> pd.Series:
    """
    Ricalcola le feature derivate per mantenere il pacchetto avversario
    fisicamente realistico e coerente con lo stack TCP/IP.
    """
    adv_sample = sample.copy()
    # 1. Coerenza del QoS (ip.dsfield <- ip.dsfield.dscp)
    if "ip.dsfield.dscp" in adv_sample and "ip.dsfield" in adv_sample:
        # Recuperiamo i 2 bit originali di ECN (se presenti)
        orig_dsfield = int(str(orig_sample["ip.dsfield"]), 16) if pd.notnull(orig_sample["ip.dsfield"]) else 0
        orig_ecn = orig_dsfield & 0x03

        new_dscp = int(adv_sample["ip.dsfield.dscp"])
        # dsfield = (dscp * 4) + ecn
        adv_sample["ip.dsfield"] = hex((new_dscp << 2) | orig_ecn)

    # 2. Coerenza dei Flag IP (ip.flags <- ip.flags.df)
    if "ip.flags.df" in adv_sample and "ip.flags" in adv_sample:
        df_bit = int(adv_sample["ip.flags.df"])
        # Standard IPv4: bit DF è 0x02. (Ignoriamo MF per semplicità, nei pacchetti PFCP non si frammenta quasi mai)
        adv_sample["ip.flags"] = hex(0x02) if df_bit == 1 else hex(0x00)

    # 3. Coerenza Porte UDP (udp.port <- udp.srcport)
    if "udp.srcport" in adv_sample and "udp.port" in adv_sample:
        adv_sample["udp.port"] = adv_sample["udp.srcport"]

    return adv_sample


ATTACK_TYPE_MAP = {
    0: "flooding",
    1: "session_deletion",
    2: "session_modification",
    5: "upf_pdn0_fault",
    6: "restoration_teid",
}

ATTACK_FEATURES = {
    "flooding": ["pfcp.msg_type"],
    "session_deletion": ["pfcp.msg_type"],
    "session_modification": ["pfcp.msg_type", "pfcp.seid"],
    "upf_pdn0_fault": [
        "pfcp.node_id_ipv4",
        "pfcp.pdr_id",
        "pfcp.f_teid_flags.ch",
        "pfcp.f_teid_flags.ch_id",
        "pfcp.f_teid_flags.v6",
    ],
    "restoration_teid": ["pfcp.f_teid.teid", "pfcp.pdr_id"],
}

# ---------------------------------------------------------------------
# MAPPATURA FEATURE MODIFICABILI
# ---------------------------------------------------------------------
FEAT_MAPPING: Dict[str, Dict[str, Any]] = {
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


def rand_bool() -> bool:
    return True if random.random() < 0.5 else False

def rand_hex(min_val, max_val) -> str:
    value = random.randint(min_val, max_val)
    hex_len = (max_val.bit_length() + 3) // 4 if max_val > 0 else 1
    return f"0x{value:0{hex_len}X}"

def rand_int(min_val, max_val) -> int:
    return random.randint(min_val, max_val)

def rand_float(min_val, max_val) -> float:
    return random.uniform(min_val, max_val)

def rand_float_int(min_val, max_val) -> float:
    return float(random.randint(min_val, max_val))

def rand_ipv4() -> str:
    while True:
        ip_int = random.randint(1, 0xFFFFFFFF - 1)
        ip_addr = ipaddress.IPv4Address(ip_int)
        if not (
            ip_addr.is_multicast
            or ip_addr.is_reserved
            or ip_addr.is_loopback
            or ip_addr.is_unspecified
            or ip_addr.is_link_local
        ):
            return str(ip_addr)

def generate_random_value(mapping: dict) -> Any:
    field_type = mapping["type"]
    if field_type == "ipv4": return rand_ipv4()
    if field_type == "bool_str": return rand_bool()
    if field_type == "hex": return rand_hex(mapping["min"], mapping["max"])
    if field_type == "int": return rand_int(mapping["min"], mapping["max"])
    if field_type == "float": return rand_float(mapping["min"], mapping["max"])
    if field_type == "float_int": return rand_float_int(mapping["min"], mapping["max"])
    return None

def random_attack(
    sample: pd.Series, attack_type: int, features_to_attack: List[str], seed: int = 42
) -> pd.Series:
    """Apply a random attack ONLY to the specified top SHAP features."""
    if seed is not None:
        random.seed(seed)

    adv_sample = sample.copy()

    for field in features_to_attack:
        mapping = FEAT_MAPPING[field]

        # Se la feature non è presente nel dataset o nei bounds, skippa
        if field not in adv_sample.index or ("min" not in mapping and mapping["type"] not in ["ipv4", "bool_str"]):
            continue

        # Skip features essential for the attack semantics
        if field in ATTACK_FEATURES.get(ATTACK_TYPE_MAP.get(attack_type, ""), []):
            continue

        value = generate_random_value(mapping)
        if value is not None:
            adv_sample[field] = value

    adv_sample = _enforce_network_constraints(adv_sample, sample)
    return adv_sample


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Random feature-space attack guided by SHAP for RAW network traffic classifiers."
    )
    argparser.add_argument(
        "--model-name", type=str, required=True, help="Name of the trained model to attack."
    )
    argparser.add_argument(
        "--ds-path", type=str, default="data/datasets/attack_dataset_raw.csv", help="The path to the attacks dataset file"
    )
    argparser.add_argument(
        "--top-k", type=int, default=10, help="Number of top SHAP modifiable features to attack (default: 10)"
    )
    args = argparser.parse_args()

    dataset = pd.read_csv(args.ds_path, sep=";", low_memory=False)
    labels = dataset["ip.opt.time_stamp"].copy()
    dataset = dataset.drop(columns=["ip.opt.time_stamp"])

    # 1. Recupero dei file SHAP fisso per la versione RAW
    shap_path = Path(
        __file__).parent.parent / f"results_raw/without_scaler/explainability/shap_features_{args.model_name}_raw.json"

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

    print(f"\n--- SHAP GUIDED ATTACK (RAW) ---")
    print(f"Modello: {args.model_name}")
    print(f"Top {args.top_k} feature modificabili estratte da SHAP:")
    for i, f in enumerate(top_k_features, 1):
        print(f"  {i}. {f}")
    print("-" * 32 + "\n")

    if not top_k_features:
        print("Nessuna feature modificabile trovata nei risultati SHAP. Esco.")
        sys.exit(1)

    # 3. Inizializzazione Dinamica dei Range SOLO per le Top K
    print("Inizializzazione dinamica dei bounds per le Top K feature...")
    for feat in top_k_features:
        mapping = FEAT_MAPPING[feat]
        if feat in dataset.columns and mapping["type"] not in ["ipv4", "bool_str"]:
            valid_data = dataset[feat].dropna()
            if not valid_data.empty:
                mapping["min"] = float(valid_data.min()) if mapping["type"] == "float" else int(valid_data.min())
                mapping["max"] = float(valid_data.max()) if mapping["type"] == "float" else int(valid_data.max())
            else:
                mapping["min"] = 0
                mapping["max"] = 100

                # 4. Configurazione Path Modello e Risultati (fissi su RAW)
    model_path = Path(__file__).parent.parent / f"data/trained_models_raw/without_scaler/{args.model_name}.pkl"
    results_path = Path(
        __file__).parent.parent / f"results_raw/without_scaler/random_attack/{args.model_name}_top{args.top_k}.json"

    detector = joblib.load(model_path)

    if results_path.exists():
        with results_path.open("r") as f:
            results = json.load(f)
    else:
        results = {}
        results_path.parent.mkdir(parents=True, exist_ok=True)

    # 5. Esecuzione Attacco
    for idx, sample in dataset.iterrows():
        if str(idx) in results:
            continue

        y_pred = detector.predict(pd.DataFrame([sample]))[0]
        init_score = detector.decision_function(pd.DataFrame([sample]))[0]

        print(f"Sample {idx} - Orig score: {init_score}")

        if y_pred == 0:
            print(f"Sample {idx} is already misclassified. Skipping attack.\n")
            results[idx] = {
                "original_score": init_score,
                "attack_type": int(labels.loc[idx]),
                "attacked_score": None,
                "success": False,
            }
            continue

        # Passiamo la lista top_k_features all'attacco
        adv_sample = random_attack(sample, int(labels.loc[idx]), top_k_features)
        adv_score = detector.decision_function(pd.DataFrame([adv_sample]))[0]

        if hasattr(detector, "_detector"):
            if adv_score < detector._detector.threshold_:
                print(f"Sample {idx} - Adv score: {adv_score} -> SUCCESS!\n")
                results[idx] = {
                    "original_score": init_score,
                    "attacked_score": adv_score,
                    "attack_type": int(labels.loc[idx]),
                    "success": True,
                }
            else:
                print(f"Sample {idx} - Adv score: {adv_score} -> FAILED\n")
                results[idx] = {
                    "original_score": init_score,
                    "attacked_score": adv_score,
                    "attack_type": int(labels.loc[idx]),
                    "success": False,
                }
        else:
            y_pred = detector.predict(pd.DataFrame([sample]))
            if y_pred == 0:
                print("Sample is already classified as benign. Skipping attack.")
            else:
                print(f"Sample {idx} - Adv score: {adv_score} -> FAILED\n")
                results[idx] = {
                    "original_score": init_score,
                    "attacked_score": adv_score,
                    "attack_type": int(labels.loc[idx]),
                    "success": False,
                }

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)