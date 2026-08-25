"""
dataset_loader.py
-----------------
Loaders for CICIDS2017 and UNSW-NB15 intrusion detection datasets.

Usage:
    python src/data_processing/dataset_loader.py --dataset cicids2017 --output data/processed/
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# CICIDS 2017
# ---------------------------------------------------------------------------

CICIDS_LABEL_MAP = {
    "BENIGN": 0,
    "PortScan": 1,
    "FTP-Patator": 2,
    "SSH-Patator": 2,
    "DoS slowloris": 3,
    "DoS Slowhttptest": 3,
    "DoS Hulk": 3,
    "DoS GoldenEye": 3,
    "Heartbleed": 3,
    "Web Attack – Brute Force": 2,
    "Web Attack – XSS": 2,
    "Web Attack – Sql Injection": 2,
    "Infiltration": 4,
    "Bot": 4,
    "DDoS": 3,
}

# Regime semantic mapping (used in plots and reports)
REGIME_NAMES = {
    0: "Normal",
    1: "Reconnaissance / Scanning",
    2: "Intrusion Attempts",
    3: "DoS / DDoS Attack",
    4: "Lateral Movement / Exfiltration",
}

CICIDS_FEATURE_COLS = [
    ' Destination Port',
    ' Flow Duration',
    ' Total Fwd Packets',
    ' Total Backward Packets',
    'Total Length of Fwd Packets',
    ' Total Length of Bwd Packets',
    ' Fwd Packet Length Max',
    ' Fwd Packet Length Min',
    ' Fwd Packet Length Mean',
    ' Fwd Packet Length Std',
    'Bwd Packet Length Max',
    ' Bwd Packet Length Min',
    ' Bwd Packet Length Mean',
    ' Bwd Packet Length Std',
    'Flow Bytes/s',            
    ' Flow Packets/s',
    ' Flow IAT Mean',
    ' Flow IAT Std',
    ' Flow IAT Max',
    ' Flow IAT Min',
    'Fwd IAT Total',
    ' Fwd IAT Mean',
    ' Fwd IAT Std',
    ' Fwd IAT Max',
    ' Fwd IAT Min',
    'Bwd IAT Total',
    ' Bwd IAT Mean',
    ' Bwd IAT Std',
    ' Bwd IAT Max',
    ' Bwd IAT Min',
    'Fwd PSH Flags',
    ' Fwd URG Flags',
    ' Fwd Header Length',
    ' Bwd Header Length',
    'Fwd Packets/s',
    ' Bwd Packets/s',
    ' Min Packet Length',
    ' Max Packet Length',
    ' Packet Length Mean',
    ' Packet Length Std',
    ' Packet Length Variance',
    ' SYN Flag Count',
    ' RST Flag Count',
    ' PSH Flag Count',
    ' ACK Flag Count',
    ' URG Flag Count',
    ' CWE Flag Count',
    ' ECE Flag Count',
    ' Down/Up Ratio',
    ' Average Packet Size',
    ' Avg Fwd Segment Size',
    ' Avg Bwd Segment Size',
]


def load_cicids2017(data_dir: str, sample_size: int = 50000) -> tuple:
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}.\n"
            "Download CICIDS2017 from: https://www.unb.ca/cic/datasets/ids-2017.html"
        )

    dfs = []
    for f in csv_files:
        print(f"  Loading {f.name} ...")
        df = pd.read_csv(f, low_memory=False)
        # Strip ALL column names — removes all leading/trailing spaces
        df.columns = df.columns.str.strip()
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows loaded: {len(df):,}")

    # Strip feature list too — match stripped column names
    FEATURES_STRIPPED = [c.strip() for c in CICIDS_FEATURE_COLS]

    # Map labels
    label_col = "Label"   # after strip(), always "Label" — no space
    df["regime"] = df[label_col].str.strip().map(CICIDS_LABEL_MAP).fillna(0).astype(int)

    # Select available features
    available_cols = [c for c in FEATURES_STRIPPED if c in df.columns]
    print(f"Matched {len(available_cols)} of {len(FEATURES_STRIPPED)} features")

    df = df[available_cols + ["regime"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    X = df[available_cols].values.astype(np.float32)
    y = df["regime"].values.astype(np.int32)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Regime distribution: { {k: int((y==k).sum()) for k in np.unique(y)} }")

    return X, y


# ---------------------------------------------------------------------------
# UNSW-NB15
# ---------------------------------------------------------------------------

UNSW_LABEL_MAP = {
    "Normal": 0,
    "Reconnaissance": 1,
    "Backdoors": 2,
    "DoS": 3,
    "Exploits": 2,
    "Analysis": 1,
    "Fuzzers": 1,
    "Worms": 4,
    "Shellcode": 4,
    "Generic": 3,
}

UNSW_FEATURE_COLS = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload",
    "sloss", "dloss", "sinpkt", "dinpkt", "sjit",
    "djit", "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat", "smean", "dmean",
    "trans_depth", "response_body_len", "ct_srv_src",
    "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
    "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]


def load_unswnb15(data_dir: str, sample_size: int = 50000) -> tuple:
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob("UNSW*.csv")) or list(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}.")

    # Official UNSW-NB15 column names (no header in CSVs)
    col_names = [
        "srcip", "sport", "dstip", "dsport", "proto", "state",
        "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
        "service", "sload", "dload", "spkts", "dpkts", "swin", "dwin",
        "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth",
        "response_body_len", "sjit", "djit", "stime", "ltime",
        "sinpkt", "dinpkt", "tcprtt", "synack", "ackdat",
        "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
        "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
        "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
        "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label"
    ]

    dfs = []
    for f in csv_files:
        print(f"  Loading {f.name} ...")
        try:
            df = pd.read_csv(f, header=None, names=col_names, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not load {f.name}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows loaded: {len(df):,}")

    # Use numeric label column (0=normal, 1=attack)
    df["regime"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    # Use numeric feature columns only
    UNSW_FEATURES = [
        "dur", "spkts", "dpkts", "sbytes", "dbytes",
        "sttl", "dttl", "sloss", "dloss", "sload", "dload",
        "sinpkt", "dinpkt", "sjit", "djit", "swin", "dwin",
        "tcprtt", "synack", "ackdat", "smeansz", "dmeansz",
        "trans_depth", "response_body_len", "ct_srv_src",
        "ct_state_ttl", "ct_dst_ltm", "ct_src_ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
        "is_ftp_login", "ct_ftp_cmd", "ct_srv_dst",
        "is_sm_ips_ports", "ct_flw_http_mthd",
    ]

    available_cols = [c for c in UNSW_FEATURES if c in df.columns]
    print(f"Matched {len(available_cols)} features")

    df = df[available_cols + ["regime"]].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    X = df[available_cols].values.astype(np.float32)
    y = df["regime"].values.astype(np.int32)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Regime distribution: { {k: int((y==k).sum()) for k in np.unique(y)} }")

    return X, y


# ---------------------------------------------------------------------------
# Save / Load processed data
# ---------------------------------------------------------------------------

def save_processed(X: np.ndarray, y: np.ndarray, output_dir: str, name: str):
    """Save processed arrays to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{name}_features.npy", X)
    np.save(output_dir / f"{name}_labels.npy", y)
    print(f"Saved to {output_dir / name}_*.npy")


def load_processed(data_dir: str, name: str) -> tuple:
    """Load preprocessed numpy arrays."""
    data_dir = Path(data_dir)
    X = np.load(data_dir / f"{name}_features.npy")
    y = np.load(data_dir / f"{name}_labels.npy")
    return X, y


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and preprocess network intrusion datasets")
    parser.add_argument("--dataset", choices=["cicids2017", "unswnb15"], required=True)
    parser.add_argument("--input", default="data/raw/", help="Path to raw dataset directory")
    parser.add_argument("--output", default="data/processed/", help="Path to save processed data")
    parser.add_argument("--sample", type=int, default=50000, help="Number of samples to load")
    args = parser.parse_args()

    print(f"\nLoading dataset: {args.dataset}")
    print(f"Input directory: {args.input}")

    if args.dataset == "cicids2017":
        X, y = load_cicids2017(args.input, sample_size=args.sample)
        save_processed(X, y, args.output, "cicids2017")
    elif args.dataset == "unswnb15":
        X, y = load_unswnb15(args.input, sample_size=args.sample)
        save_processed(X, y, args.output, "unswnb15")

    print("\nDone.")
