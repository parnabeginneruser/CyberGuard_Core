"""
================================================================================
                                CYBERGUARD
       Dual-Path AI Network Intrusion Detection & Threat Intelligence Engine
================================================================================

Architecture Overview:
----------------------
CyberGuard deploys a synchronized dual-path defense architecture:

  [ Path A: Known Attack Classifier (XGBoost) ]
  - Supervised gradient-boosted decision trees.
  - High-precision identification of known network threats (DoS, PortScan, BruteForce, Botnet, etc.).
  - Rapid multi-class threat categorization with probability estimation.

  [ Path B: Zero-Day Anomaly Detector (Deep Keras Autoencoder) ]
  - Unsupervised deep learning autoencoder trained exclusively on benign baseline traffic.
  - Learns the latent manifold of normal network telemetry.
  - Flags novel zero-day exploits & evasive attacks via reconstruction error thresholding.

  [ Hybrid Threat Triage Engine ]
  - Correlates Path A signature classification with Path B reconstruction metrics.
  - Detects stealth/zero-day attacks that bypass supervised classifiers while triaging known signatures.

Author: CyberGuard AI Engineering Team
Version: 1.0.0
================================================================================
"""

import os
import sys
import json
import time
import argparse
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# Suppress TensorFlow informational logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers

import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CyberGuard")


# ============================================================================
# Dataclasses & Configuration
# ============================================================================

@dataclass
class ThreatVerdict:
    """Represents the unified hybrid threat verdict for a network flow."""
    sample_id: int
    status: str  # "BENIGN", "KNOWN_ATTACK", "SUSPECTED_ZERO_DAY", "LOW_CONF_ALERT"
    attack_type: str
    threat_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    path_a_prediction: str
    path_a_confidence: float
    path_b_recon_error: float
    path_b_threshold: float
    path_b_is_anomaly: bool
    anomaly_ratio: float  # recon_error / threshold
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Synthetic Traffic Generator
# ============================================================================

class SyntheticTrafficGenerator:
    """
    Generates realistic, multi-feature network flow datasets mimicking modern
    intrusion detection benchmarks (e.g., CIC-IDS / NSL-KDD).
    Includes normal baseline flows, known attacks (Path A), and novel zero-day
    anomalies (Path B).
    """

    FEATURE_COLUMNS = [
        "flow_duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "packet_rate",
        "byte_rate",
        "failed_logins",
        "logged_in",
        "count",
        "srv_count",
        "serror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_src_port_rate",
    ]

    NUMERIC_FEATURES = [
        "flow_duration",
        "src_bytes",
        "dst_bytes",
        "packet_rate",
        "byte_rate",
        "failed_logins",
        "logged_in",
        "count",
        "srv_count",
        "serror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_src_port_rate",
    ]

    CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

    KNOWN_ATTACK_TYPES = [
        "DoS_SYN_Flood",
        "DDoS_HTTP_Flood",
        "Port_Scan",
        "Brute_Force_SSH",
        "Botnet_C2",
    ]

    ZERO_DAY_ATTACK_TYPES = [
        "ZeroDay_Polymorphic_Exfiltration",
        "ZeroDay_AI_Evasion_Tunnel",
    ]

    @classmethod
    def generate_dataset(
        cls,
        n_samples: int = 15000,
        benign_ratio: float = 0.65,
        include_zero_days: bool = True,
        zero_day_ratio: float = 0.08,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Generate synthetic telemetry data with benign, known attacks, and zero-day threats.
        """
        np.random.seed(random_state)
        data = []

        n_benign = int(n_samples * benign_ratio)
        n_zero_day = int(n_samples * zero_day_ratio) if include_zero_days else 0
        n_known = n_samples - n_benign - n_zero_day

        # --------------------------------------------------------------------
        # 1. Benign / Normal Traffic
        # --------------------------------------------------------------------
        for _ in range(n_benign):
            proto = np.random.choice(["TCP", "UDP", "ICMP"], p=[0.75, 0.20, 0.05])
            serv = np.random.choice(["http", "ssl", "dns", "ssh", "smtp"], p=[0.45, 0.30, 0.15, 0.05, 0.05])
            flag = np.random.choice(["SF", "S0", "REJ"], p=[0.92, 0.05, 0.03])
            
            flow_dur = np.random.exponential(scale=12.0) + 0.1
            src_b = int(np.random.gamma(shape=3.0, scale=350.0)) + 40
            dst_b = int(np.random.gamma(shape=5.0, scale=1200.0)) + 40
            pkt_rate = float(np.random.normal(loc=18.0, scale=5.0))
            pkt_rate = max(0.5, pkt_rate)
            byte_rate = (src_b + dst_b) / max(flow_dur, 0.1)

            rec = {
                "flow_duration": round(flow_dur, 4),
                "protocol_type": proto,
                "service": serv,
                "flag": flag,
                "src_bytes": src_b,
                "dst_bytes": dst_b,
                "packet_rate": round(pkt_rate, 2),
                "byte_rate": round(byte_rate, 2),
                "failed_logins": 0,
                "logged_in": 1 if serv in ["http", "ssl", "ssh"] and np.random.rand() > 0.1 else 0,
                "count": int(np.random.poisson(lam=4)),
                "srv_count": int(np.random.poisson(lam=4)),
                "serror_rate": round(float(np.random.beta(0.5, 20.0)), 4),
                "same_srv_rate": round(float(np.random.beta(15.0, 1.0)), 4),
                "diff_srv_rate": round(float(np.random.beta(0.5, 10.0)), 4),
                "dst_host_count": int(np.random.randint(1, 100)),
                "dst_host_srv_count": int(np.random.randint(1, 100)),
                "dst_host_same_src_port_rate": round(float(np.random.beta(1.0, 5.0)), 4),
                "label": "BENIGN",
                "is_attack": 0,
            }
            data.append(rec)

        # --------------------------------------------------------------------
        # 2. Known Attacks (Path A Target Classes)
        # --------------------------------------------------------------------
        attack_types = cls.KNOWN_ATTACK_TYPES
        for _ in range(n_known):
            atk = np.random.choice(attack_types)

            if atk == "DoS_SYN_Flood":
                rec = {
                    "flow_duration": round(float(np.random.exponential(0.5) + 0.01), 4),
                    "protocol_type": "TCP",
                    "service": "http",
                    "flag": "S0",
                    "src_bytes": int(np.random.normal(60, 5)),
                    "dst_bytes": 0,
                    "packet_rate": round(float(np.random.normal(850.0, 120.0)), 2),
                    "byte_rate": round(float(np.random.normal(45000.0, 8000.0)), 2),
                    "failed_logins": 0,
                    "logged_in": 0,
                    "count": int(np.random.randint(250, 512)),
                    "srv_count": int(np.random.randint(200, 512)),
                    "serror_rate": round(float(np.random.uniform(0.90, 1.0)), 4),
                    "same_srv_rate": round(float(np.random.uniform(0.85, 1.0)), 4),
                    "diff_srv_rate": round(float(np.random.uniform(0.0, 0.05)), 4),
                    "dst_host_count": 255,
                    "dst_host_srv_count": int(np.random.randint(200, 255)),
                    "dst_host_same_src_port_rate": round(float(np.random.uniform(0.8, 1.0)), 4),
                    "label": atk,
                    "is_attack": 1,
                }
            elif atk == "DDoS_HTTP_Flood":
                rec = {
                    "flow_duration": round(float(np.random.uniform(0.2, 2.0)), 4),
                    "protocol_type": "TCP",
                    "service": "http",
                    "flag": "SF",
                    "src_bytes": int(np.random.normal(1200, 150)),
                    "dst_bytes": int(np.random.normal(300, 50)),
                    "packet_rate": round(float(np.random.normal(400.0, 60.0)), 2),
                    "byte_rate": round(float(np.random.normal(60000.0, 9000.0)), 2),
                    "failed_logins": 0,
                    "logged_in": 1,
                    "count": int(np.random.randint(180, 400)),
                    "srv_count": int(np.random.randint(180, 400)),
                    "serror_rate": 0.02,
                    "same_srv_rate": 0.98,
                    "diff_srv_rate": 0.02,
                    "dst_host_count": 255,
                    "dst_host_srv_count": 255,
                    "dst_host_same_src_port_rate": round(float(np.random.uniform(0.4, 0.9)), 4),
                    "label": atk,
                    "is_attack": 1,
                }
            elif atk == "Port_Scan":
                rec = {
                    "flow_duration": round(float(np.random.exponential(0.1) + 0.005), 4),
                    "protocol_type": np.random.choice(["TCP", "UDP"]),
                    "service": np.random.choice(["other", "http", "ssh", "dns"]),
                    "flag": np.random.choice(["REJ", "S0", "RSTO"]),
                    "src_bytes": int(np.random.choice([0, 40, 64])),
                    "dst_bytes": 0,
                    "packet_rate": round(float(np.random.normal(250.0, 40.0)), 2),
                    "byte_rate": round(float(np.random.normal(12000.0, 2000.0)), 2),
                    "failed_logins": 0,
                    "logged_in": 0,
                    "count": int(np.random.randint(150, 450)),
                    "srv_count": int(np.random.randint(1, 10)),
                    "serror_rate": round(float(np.random.uniform(0.7, 1.0)), 4),
                    "same_srv_rate": round(float(np.random.uniform(0.01, 0.08)), 4),
                    "diff_srv_rate": round(float(np.random.uniform(0.85, 1.0)), 4),
                    "dst_host_count": 255,
                    "dst_host_srv_count": int(np.random.randint(1, 15)),
                    "dst_host_same_src_port_rate": round(float(np.random.uniform(0.0, 0.1)), 4),
                    "label": atk,
                    "is_attack": 1,
                }
            elif atk == "Brute_Force_SSH":
                rec = {
                    "flow_duration": round(float(np.random.uniform(1.0, 6.0)), 4),
                    "protocol_type": "TCP",
                    "service": "ssh",
                    "flag": "SF",
                    "src_bytes": int(np.random.normal(850, 100)),
                    "dst_bytes": int(np.random.normal(600, 80)),
                    "packet_rate": round(float(np.random.normal(30.0, 6.0)), 2),
                    "byte_rate": round(float(np.random.normal(1800.0, 300.0)), 2),
                    "failed_logins": int(np.random.randint(5, 25)),
                    "logged_in": 0,
                    "count": int(np.random.randint(40, 120)),
                    "srv_count": int(np.random.randint(40, 120)),
                    "serror_rate": 0.05,
                    "same_srv_rate": 0.95,
                    "diff_srv_rate": 0.05,
                    "dst_host_count": int(np.random.randint(50, 150)),
                    "dst_host_srv_count": int(np.random.randint(50, 150)),
                    "dst_host_same_src_port_rate": round(float(np.random.uniform(0.1, 0.4)), 4),
                    "label": atk,
                    "is_attack": 1,
                }
            else:  # Botnet_C2
                rec = {
                    "flow_duration": round(float(np.random.exponential(45.0) + 10.0), 4),
                    "protocol_type": "TCP",
                    "service": np.random.choice(["ssl", "dns", "http"]),
                    "flag": "SF",
                    "src_bytes": int(np.random.normal(420, 60)),
                    "dst_bytes": int(np.random.normal(380, 50)),
                    "packet_rate": round(float(np.random.normal(2.5, 0.5)), 2),
                    "byte_rate": round(float(np.random.normal(120.0, 25.0)), 2),
                    "failed_logins": 0,
                    "logged_in": 1,
                    "count": int(np.random.randint(1, 12)),
                    "srv_count": int(np.random.randint(1, 12)),
                    "serror_rate": 0.0,
                    "same_srv_rate": 0.88,
                    "diff_srv_rate": 0.12,
                    "dst_host_count": int(np.random.randint(1, 20)),
                    "dst_host_srv_count": int(np.random.randint(1, 20)),
                    "dst_host_same_src_port_rate": 0.0,
                    "label": atk,
                    "is_attack": 1,
                }
            data.append(rec)

        # --------------------------------------------------------------------
        # 3. Novel Zero-Day Attacks (Unseen Patterns for Path B)
        # --------------------------------------------------------------------
        for _ in range(n_zero_day):
            zd = np.random.choice(cls.ZERO_DAY_ATTACK_TYPES)
            if zd == "ZeroDay_Polymorphic_Exfiltration":
                # High src_bytes, extremely low packet_rate, disguised over DNS/ICMP
                rec = {
                    "flow_duration": round(float(np.random.uniform(120.0, 600.0)), 4),
                    "protocol_type": np.random.choice(["UDP", "ICMP"]),
                    "service": "dns",
                    "flag": "SF",
                    "src_bytes": int(np.random.uniform(85000, 350000)),
                    "dst_bytes": int(np.random.uniform(200, 800)),
                    "packet_rate": round(float(np.random.uniform(0.1, 0.8)), 2),
                    "byte_rate": round(float(np.random.uniform(300.0, 900.0)), 2),
                    "failed_logins": 0,
                    "logged_in": 0,
                    "count": int(np.random.randint(1, 5)),
                    "srv_count": int(np.random.randint(1, 5)),
                    "serror_rate": 0.0,
                    "same_srv_rate": 0.99,
                    "diff_srv_rate": 0.01,
                    "dst_host_count": int(np.random.randint(1, 10)),
                    "dst_host_srv_count": int(np.random.randint(1, 10)),
                    "dst_host_same_src_port_rate": 0.0,
                    "label": zd,
                    "is_attack": 1,
                }
            else:  # ZeroDay_AI_Evasion_Tunnel
                # Stochastically perturbed timing and burst patterns
                rec = {
                    "flow_duration": round(float(np.random.uniform(0.01, 80.0)), 4),
                    "protocol_type": "TCP",
                    "service": "ssl",
                    "flag": np.random.choice(["SF", "SH"]),
                    "src_bytes": int(np.random.uniform(15000, 90000)),
                    "dst_bytes": int(np.random.uniform(45000, 180000)),
                    "packet_rate": round(float(np.random.exponential(75.0)), 2),
                    "byte_rate": round(float(np.random.exponential(25000.0)), 2),
                    "failed_logins": int(np.random.choice([0, 1])),
                    "logged_in": 1,
                    "count": int(np.random.randint(80, 220)),
                    "srv_count": int(np.random.randint(10, 40)),
                    "serror_rate": round(float(np.random.uniform(0.1, 0.4)), 4),
                    "same_srv_rate": round(float(np.random.uniform(0.2, 0.6)), 4),
                    "diff_srv_rate": round(float(np.random.uniform(0.3, 0.8)), 4),
                    "dst_host_count": int(np.random.randint(100, 255)),
                    "dst_host_srv_count": int(np.random.randint(20, 80)),
                    "dst_host_same_src_port_rate": round(float(np.random.uniform(0.3, 0.7)), 4),
                    "label": zd,
                    "is_attack": 1,
                }
            data.append(rec)

        df = pd.DataFrame(data)
        # Shuffle rows
        df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        return df


# ============================================================================
# Preprocessing Engine
# ============================================================================

class DataPreprocessor:
    """
    Standardizes feature encoding, numerical normalization, and ensures
    feature alignment between training and production inference.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.cat_columns: List[str] = []
        self.num_columns: List[str] = []
        self.feature_columns: List[str] = []
        self.is_fitted: bool = False
        self.classes_: List[str] = []

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = "label") -> "DataPreprocessor":
        """Fit scaler on numerical features and prepare schema."""
        df_work = df.copy()

        # Separate feature columns
        cols = [c for c in df_work.columns if c not in ["label", "is_attack"]]
        
        self.cat_columns = [c for c in cols if df_work[c].dtype == "object" or df_work[c].dtype.name == "category"]
        self.num_columns = [c for c in cols if c not in self.cat_columns]

        # One-hot encode categoricals to learn schema
        encoded_df = pd.get_dummies(df_work[cols], columns=self.cat_columns, drop_first=False)
        self.feature_columns = list(encoded_df.columns)

        # Fit standard scaler
        self.scaler.fit(encoded_df.values.astype(np.float32))

        # Fit label encoder if target provided
        if target_col and target_col in df_work.columns:
            self.label_encoder.fit(df_work[target_col].astype(str))
            self.classes_ = list(self.label_encoder.classes_)

        self.is_fitted = True
        return self

    def transform(
        self, df: pd.DataFrame, target_col: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform dataframe to normalized feature matrix and optional encoded targets."""
        if not self.is_fitted:
            raise RuntimeError("DataPreprocessor must be fitted before transforming.")

        df_work = df.copy()
        cols = [c for c in df_work.columns if c not in ["label", "is_attack"]]

        # One-hot encode
        encoded_df = pd.get_dummies(df_work[cols], columns=[c for c in self.cat_columns if c in df_work.columns])

        # Align columns with fitted schema (fill missing dummy cols with 0)
        for c in self.feature_columns:
            if c not in encoded_df.columns:
                encoded_df[c] = 0.0

        # Ensure correct column ordering
        encoded_df = encoded_df[self.feature_columns]

        # Scale features
        X = self.scaler.transform(encoded_df.values.astype(np.float32))

        y = None
        if target_col and target_col in df_work.columns:
            # Handle unknown labels in inference safely
            known_classes = set(self.label_encoder.classes_)
            labels = df_work[target_col].astype(str).values
            safe_labels = [lbl if lbl in known_classes else "BENIGN" for lbl in labels]
            y = self.label_encoder.transform(safe_labels)

        return X, y

    def fit_transform(
        self, df: pd.DataFrame, target_col: Optional[str] = "label"
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit and transform simultaneously."""
        self.fit(df, target_col)
        return self.transform(df, target_col)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save preprocessor state."""
        payload = {
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "cat_columns": self.cat_columns,
            "num_columns": self.num_columns,
            "feature_columns": self.feature_columns,
            "classes_": self.classes_,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, filepath)
        logger.info(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DataPreprocessor":
        """Load preprocessor state from disk."""
        payload = joblib.load(filepath)
        obj = cls()
        obj.scaler = payload["scaler"]
        obj.label_encoder = payload["label_encoder"]
        obj.cat_columns = payload["cat_columns"]
        obj.num_columns = payload["num_columns"]
        obj.feature_columns = payload["feature_columns"]
        obj.classes_ = payload["classes_"]
        obj.is_fitted = payload["is_fitted"]
        return obj


# ============================================================================
# Path A: Known Attack Classifier (XGBoost)
# ============================================================================

class KnownAttackClassifier:
    """
    Path A: Supervised Multi-Class XGBoost Classifier.
    Trained on labeled network flows to detect and classify known attack types.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.08,
        max_depth: int = 6,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        random_state: int = 42,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "n_jobs": -1,
        }
        self.model: Optional[xgb.XGBClassifier] = None
        self.class_names: List[str] = []
        self.is_trained: bool = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Train the XGBoost Classifier on known network attacks."""
        logger.info("Initializing Path A: Training XGBoost Known Attack Classifier...")
        start_time = time.time()

        n_classes = len(np.unique(y_train))
        objective = "binary:logistic" if n_classes <= 2 else "multi:softprob"

        self.model = xgb.XGBClassifier(
            objective=objective,
            num_class=n_classes if n_classes > 2 else None,
            **self.params,
        )

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        self.class_names = class_names if class_names else [f"Class_{i}" for i in range(n_classes)]
        self.is_trained = True
        elapsed = time.time() - start_time
        logger.info(f"Path A (XGBoost) training completed in {elapsed:.2f}s across {n_classes} target classes.")

        # Compute training accuracy
        train_preds = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)

        results = {
            "training_time_seconds": round(elapsed, 3),
            "train_accuracy": round(float(train_acc), 4),
            "num_classes": n_classes,
            "class_names": self.class_names,
        }

        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_preds)
            results["val_accuracy"] = round(float(val_acc), 4)
            logger.info(f"Path A Validation Accuracy: {val_acc * 100:.2f}%")

        return results

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Run comprehensive classification evaluation."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Path A Model is not trained yet.")

        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)

        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
        
        report_dict = classification_report(
            y_test,
            preds,
            target_names=self.class_names if len(self.class_names) == len(np.unique(y_test)) else None,
            output_dict=True,
            zero_division=0,
        )
        conf_mat = confusion_matrix(y_test, preds).tolist()

        return {
            "accuracy": round(float(acc), 4),
            "precision_weighted": round(float(prec), 4),
            "recall_weighted": round(float(rec), 4),
            "f1_weighted": round(float(f1), 4),
            "classification_report": report_dict,
            "confusion_matrix": conf_mat,
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            - predicted_class_indices: np.ndarray
            - predicted_class_names: np.ndarray
            - confidence_scores: np.ndarray (probability of predicted class)
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Path A Model is not trained yet.")

        probs = self.model.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        
        pred_names = np.array([
            self.class_names[idx] if idx < len(self.class_names) else "UNKNOWN"
            for idx in pred_indices
        ])

        return pred_indices, pred_names, confidences

    def get_feature_importances(self, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Return sorted feature importances."""
        if not self.is_trained or self.model is None:
            return []
        importances = self.model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        return [
            {"feature": feature_names[i], "importance": round(float(importances[i]), 5)}
            for i in sorted_indices
        ]

    def save(self, model_dir: Union[str, Path]) -> None:
        """Serialize XGBoost model and class metadata."""
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        model_file = path / "xgboost_model.json"
        meta_file = path / "path_a_metadata.json"

        self.model.save_model(str(model_file))
        with open(meta_file, "w") as f:
            json.dump({
                "class_names": self.class_names,
                "params": self.params,
                "is_trained": self.is_trained,
            }, f, indent=2)
        logger.info(f"Path A artifacts saved to {path}")

    def load(self, model_dir: Union[str, Path]) -> None:
        """Load XGBoost model and metadata."""
        path = Path(model_dir)
        model_file = path / "xgboost_model.json"
        meta_file = path / "path_a_metadata.json"

        with open(meta_file, "r") as f:
            meta = json.load(f)
        self.class_names = meta["class_names"]
        self.params = meta["params"]
        self.is_trained = meta["is_trained"]

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_file))
        logger.info(f"Path A model loaded successfully from {path}")


# ============================================================================
# Path B: Zero-Day Anomaly Detector (Keras Deep Autoencoder)
# ============================================================================

class ZeroDayAutoencoder:
    """
    Path B: Unsupervised Deep Autoencoder for Zero-Day Network Threat Detection.
    Trained strictly on normal/benign network traffic manifolds.
    Calculates reconstruction error and flags anomalies exceeding dynamic statistical thresholds.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        latent_dim: int = 8,
        hidden_layers: Tuple[int, ...] = (64, 32, 16),
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.model: Optional[keras.Model] = None
        self.threshold: float = 0.0
        self.mean_benign_loss: float = 0.0
        self.std_benign_loss: float = 0.0
        self.is_trained: bool = False

    def build_architecture(self, input_dim: int) -> keras.Model:
        """
        Constructs a symmetric deep autoencoder with batch normalization,
        leaky activations, and residual compression.
        """
        self.input_dim = input_dim
        
        # --- Encoder ---
        inputs = layers.Input(shape=(input_dim,), name="flow_features_input")
        x = inputs
        for idx, units in enumerate(self.hidden_layers):
            x = layers.Dense(
                units,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-5),
                name=f"encoder_dense_{idx + 1}_{units}",
            )(x)
            x = layers.BatchNormalization(name=f"encoder_bn_{idx + 1}")(x)
            x = layers.LeakyReLU(negative_slope=0.1, name=f"encoder_leaky_{idx + 1}")(x)
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate, name=f"encoder_drop_{idx + 1}")(x)

        # --- Latent Bottleneck ---
        latent = layers.Dense(
            self.latent_dim,
            activation="linear",
            kernel_initializer="he_normal",
            name="latent_bottleneck",
        )(x)

        # --- Decoder ---
        x = latent
        for idx, units in enumerate(reversed(self.hidden_layers)):
            x = layers.Dense(
                units,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-5),
                name=f"decoder_dense_{idx + 1}_{units}",
            )(x)
            x = layers.BatchNormalization(name=f"decoder_bn_{idx + 1}")(x)
            x = layers.LeakyReLU(negative_slope=0.1, name=f"decoder_leaky_{idx + 1}")(x)

        # --- Reconstruction Output ---
        outputs = layers.Dense(
            input_dim,
            activation="linear",
            name="reconstruction_output",
        )(x)

        autoencoder = keras.Model(inputs=inputs, outputs=outputs, name="CyberGuard_DeepAutoencoder")
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        autoencoder.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return autoencoder

    def train(
        self,
        X_benign_train: np.ndarray,
        X_benign_val: Optional[np.ndarray] = None,
        epochs: int = 40,
        batch_size: int = 64,
        threshold_percentile: float = 98.5,
    ) -> Dict[str, Any]:
        """
        Train Autoencoder strictly on benign normal traffic and calculate anomaly threshold.
        """
        logger.info("Initializing Path B: Training Deep Autoencoder on Benign Baseline...")
        input_dim = X_benign_train.shape[1]
        self.model = self.build_architecture(input_dim)

        training_callbacks = [
            callbacks.EarlyStopping(
                monitor="val_loss" if X_benign_val is not None else "loss",
                patience=8,
                restore_best_weights=True,
                verbose=0,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_benign_val is not None else "loss",
                factor=0.5,
                patience=4,
                min_lr=1e-5,
                verbose=0,
            ),
        ]

        val_data = (X_benign_val, X_benign_val) if X_benign_val is not None else None

        start_time = time.time()
        history = self.model.fit(
            X_benign_train,
            X_benign_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=training_callbacks,
            shuffle=True,
            verbose=0,
        )
        elapsed = time.time() - start_time

        # Calculate anomaly detection threshold
        eval_data = X_benign_val if X_benign_val is not None else X_benign_train
        recon_errors = self.compute_reconstruction_errors(eval_data)
        
        self.mean_benign_loss = float(np.mean(recon_errors))
        self.std_benign_loss = float(np.std(recon_errors))
        
        # Dual-check threshold: percentile or (mean + 3 * std)
        percentile_thresh = float(np.percentile(recon_errors, threshold_percentile))
        std_thresh = self.mean_benign_loss + (3.0 * self.std_benign_loss)
        self.threshold = float(max(percentile_thresh, std_thresh))

        self.is_trained = True
        logger.info(
            f"Path B (Autoencoder) trained in {elapsed:.2f}s. "
            f"Baseline MSE Mean={self.mean_benign_loss:.5f}, Std={self.std_benign_loss:.5f}. "
            f"Anomaly Threshold={self.threshold:.5f} (p{threshold_percentile})"
        )

        return {
            "training_time_seconds": round(elapsed, 3),
            "epochs_trained": len(history.history["loss"]),
            "final_loss": round(float(history.history["loss"][-1]), 6),
            "anomaly_threshold": round(self.threshold, 6),
            "mean_benign_loss": round(self.mean_benign_loss, 6),
            "std_benign_loss": round(self.std_benign_loss, 6),
        }

    def compute_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Compute sample-wise Mean Squared Error reconstruction loss."""
        if not self.is_trained and self.model is None:
            raise RuntimeError("Path B Autoencoder is not initialized or trained.")
        
        reconstructions = self.model.predict(X, verbose=0, batch_size=256)
        # Compute MSE per row
        mse_errors = np.mean(np.square(X - reconstructions), axis=1)
        return mse_errors

    def predict_anomalies(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect zero-day anomalies based on reconstruction error.
        Returns:
            - is_anomaly: boolean array (True if recon_error > threshold)
            - reconstruction_errors: np.ndarray
            - anomaly_ratios: recon_error / threshold
        """
        errors = self.compute_reconstruction_errors(X)
        is_anomaly = errors > self.threshold
        ratios = errors / max(self.threshold, 1e-8)
        return is_anomaly, errors, ratios

    def evaluate_zero_day_detection(
        self,
        X_benign_test: np.ndarray,
        X_zero_day_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate Path B specifically on zero-day novel attacks vs benign traffic.
        """
        benign_errors = self.compute_reconstruction_errors(X_benign_test)
        zero_day_errors = self.compute_reconstruction_errors(X_zero_day_test)

        benign_fp = np.sum(benign_errors > self.threshold)
        benign_fpr = benign_fp / len(X_benign_test)

        zero_day_tp = np.sum(zero_day_errors > self.threshold)
        zero_day_tpr = zero_day_tp / len(X_zero_day_test)

        # Combine for AUC-ROC calculation
        y_true = np.concatenate([np.zeros(len(X_benign_test)), np.ones(len(X_zero_day_test))])
        y_scores = np.concatenate([benign_errors, zero_day_errors])
        auc_roc = roc_auc_score(y_true, y_scores)

        return {
            "zero_day_detection_rate_tpr": round(float(zero_day_tpr), 4),
            "false_positive_rate_fpr": round(float(benign_fpr), 4),
            "auc_roc": round(float(auc_roc), 4),
            "benign_test_samples": len(X_benign_test),
            "zero_day_test_samples": len(X_zero_day_test),
            "mean_benign_error": round(float(np.mean(benign_errors)), 5),
            "mean_zero_day_error": round(float(np.mean(zero_day_errors)), 5),
        }

    def save(self, model_dir: Union[str, Path]) -> None:
        """Save Keras model architecture/weights and anomaly threshold metadata."""
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        model_file = path / "autoencoder_model.keras"
        meta_file = path / "path_b_metadata.json"

        self.model.save(str(model_file))
        with open(meta_file, "w") as f:
            json.dump({
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "hidden_layers": list(self.hidden_layers),
                "dropout_rate": self.dropout_rate,
                "learning_rate": self.learning_rate,
                "threshold": self.threshold,
                "mean_benign_loss": self.mean_benign_loss,
                "std_benign_loss": self.std_benign_loss,
                "is_trained": self.is_trained,
            }, f, indent=2)
        logger.info(f"Path B artifacts saved to {path}")

    def load(self, model_dir: Union[str, Path]) -> None:
        """Load Keras Autoencoder model and metadata."""
        path = Path(model_dir)
        model_file = path / "autoencoder_model.keras"
        meta_file = path / "path_b_metadata.json"

        with open(meta_file, "r") as f:
            meta = json.load(f)

        self.input_dim = meta["input_dim"]
        self.latent_dim = meta["latent_dim"]
        self.hidden_layers = tuple(meta["hidden_layers"])
        self.dropout_rate = meta["dropout_rate"]
        self.learning_rate = meta["learning_rate"]
        self.threshold = meta["threshold"]
        self.mean_benign_loss = meta["mean_benign_loss"]
        self.std_benign_loss = meta["std_benign_loss"]
        self.is_trained = meta["is_trained"]

        self.model = keras.models.load_model(str(model_file))
        logger.info(f"Path B autoencoder model loaded successfully from {path}")


# ============================================================================
# Hybrid Decision & Threat Triage Engine
# ============================================================================

class CyberGuardEngine:
    """
    Main orchestrator combining:
    1. Preprocessing & feature scaling
    2. Path A (XGBoost Known Attack Classifier)
    3. Path B (Keras Autoencoder Zero-Day Detector)
    4. Threat intelligence decision fusion & alert generation
    """

    def __init__(self, models_dir: Union[str, Path] = "models"):
        self.models_dir = Path(models_dir)
        self.preprocessor = DataPreprocessor()
        self.path_a = KnownAttackClassifier()
        self.path_b = ZeroDayAutoencoder()
        self.is_ready: bool = False

    def train_pipeline(
        self,
        df_train: pd.DataFrame,
        val_split: float = 0.20,
        epochs_ae: int = 35,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        End-to-end training:
        - Fit preprocessor
        - Train Path A on all known labeled classes
        - Train Path B strictly on Benign flows
        """
        logger.info("=" * 70)
        logger.info("STARTING CYBERGUARD DUAL-PATH TRAINING PIPELINE")
        logger.info("=" * 70)

        # 1. Separate training data
        train_df, val_df = train_test_split(
            df_train, test_size=val_split, random_state=random_state, stratify=df_train["label"]
        )

        # 2. Fit Preprocessor
        logger.info("Fitting DataPreprocessor on telemetry data...")
        self.preprocessor.fit(train_df, target_col="label")
        X_train, y_train = self.preprocessor.transform(train_df, target_col="label")
        X_val, y_val = self.preprocessor.transform(val_df, target_col="label")

        # 3. Train Path A (XGBoost on all classes)
        path_a_metrics = self.path_a.train(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            class_names=self.preprocessor.classes_,
        )

        # 4. Train Path B (Autoencoder strictly on Benign subset)
        benign_train_mask = (train_df["label"] == "BENIGN").values
        benign_val_mask = (val_df["label"] == "BENIGN").values

        X_benign_train = X_train[benign_train_mask]
        X_benign_val = X_val[benign_val_mask]

        logger.info(f"Filtering {len(X_benign_train)} benign samples for Path B baseline training...")
        path_b_metrics = self.path_b.train(
            X_benign_train,
            X_benign_val=X_benign_val,
            epochs=epochs_ae,
        )

        self.is_ready = True

        # Save pipeline artifacts automatically
        self.save()

        logger.info("=" * 70)
        logger.info("CYBERGUARD TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        return {
            "path_a": path_a_metrics,
            "path_b": path_b_metrics,
            "feature_count": len(self.preprocessor.feature_columns),
            "classes": self.preprocessor.classes_,
        }

    def analyze_traffic(self, df_traffic: pd.DataFrame) -> List[ThreatVerdict]:
        """
        Performs dual-path inference and returns structured ThreatVerdicts.
        """
        if not self.is_ready:
            self.load()

        X, _ = self.preprocessor.transform(df_traffic)

        # Run Path A
        _, path_a_preds, path_a_confs = self.path_a.predict(X)

        # Run Path B
        is_anomalies, recon_errors, anomaly_ratios = self.path_b.predict_anomalies(X)

        verdicts: List[ThreatVerdict] = []
        for i in range(len(df_traffic)):
            p_a_class = path_a_preds[i]
            p_a_conf = float(path_a_confs[i])
            r_err = float(recon_errors[i])
            is_anom = bool(is_anomalies[i])
            a_ratio = float(anomaly_ratios[i])
            thresh = float(self.path_b.threshold)

            # Decision Fusion Logic
            if p_a_class != "BENIGN" and p_a_conf >= 0.70:
                # Path A confidently identified a known attack signature
                status = "KNOWN_ATTACK"
                attack_type = p_a_class
                threat_level = "CRITICAL" if p_a_conf >= 0.90 else "HIGH"
                explanation = (
                    f"Signature matched known threat [{p_a_class}] with {p_a_conf * 100:.1f}% confidence. "
                    f"Path B reconstruction error was {r_err:.5f} (Threshold: {thresh:.5f})."
                )
            elif is_anom and (p_a_class == "BENIGN" or p_a_conf < 0.70):
                # Path A missed or had low confidence, but Path B Autoencoder flagged anomaly!
                status = "SUSPECTED_ZERO_DAY"
                attack_type = "Zero-Day / Novel Anomaly"
                threat_level = "CRITICAL" if a_ratio > 1.8 else "HIGH"
                explanation = (
                    f"POTENTIAL ZERO-DAY: Flow reconstruction error ({r_err:.5f}) exceeded normal threshold "
                    f"({thresh:.5f}) by {(a_ratio - 1.0) * 100:.1f}%. Traffic pattern deviates from benign baseline."
                )
            elif p_a_class != "BENIGN" and p_a_conf < 0.70:
                status = "LOW_CONF_ALERT"
                attack_type = f"Suspected_{p_a_class}"
                threat_level = "MEDIUM"
                explanation = (
                    f"Low confidence signature match for [{p_a_class}] ({p_a_conf * 100:.1f}%). "
                    f"Autoencoder anomaly flag: {is_anom}."
                )
            else:
                status = "BENIGN"
                attack_type = "Normal Traffic"
                threat_level = "LOW"
                explanation = f"Flow aligns with benign baseline manifold (Recon Error: {r_err:.5f} <= {thresh:.5f})."

            verdicts.append(
                ThreatVerdict(
                    sample_id=i,
                    status=status,
                    attack_type=attack_type,
                    threat_level=threat_level,
                    path_a_prediction=p_a_class,
                    path_a_confidence=round(p_a_conf, 4),
                    path_b_recon_error=round(r_err, 6),
                    path_b_threshold=round(thresh, 6),
                    path_b_is_anomaly=is_anom,
                    anomaly_ratio=round(a_ratio, 3),
                    explanation=explanation,
                )
            )

        return verdicts

    def evaluate_system(
        self,
        test_df: pd.DataFrame,
        zero_day_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive benchmark of Path A, Path B, and Hybrid Fusion on test data.
        """
        logger.info("Evaluating CyberGuard System Performance...")
        if not self.is_ready:
            self.load()

        # 1. Evaluate Path A on standard test set
        X_test, y_test = self.preprocessor.transform(test_df, target_col="label")
        path_a_results = self.path_a.evaluate(X_test, y_test)

        # 2. Evaluate Path B on Benign vs Zero-Day
        path_b_results = {}
        if zero_day_df is not None:
            benign_test = test_df[test_df["label"] == "BENIGN"]
            X_benign, _ = self.preprocessor.transform(benign_test)
            X_zd, _ = self.preprocessor.transform(zero_day_df)
            path_b_results = self.path_b.evaluate_zero_day_detection(X_benign, X_zd)

        # 3. Evaluate Hybrid Triage on Zero-Day Set
        hybrid_zd_verdicts = []
        if zero_day_df is not None:
            verdicts = self.analyze_traffic(zero_day_df)
            zd_caught = sum(1 for v in verdicts if v.status in ["SUSPECTED_ZERO_DAY", "KNOWN_ATTACK"])
            hybrid_zd_catch_rate = zd_caught / len(zero_day_df)
            hybrid_zd_verdicts = {
                "total_zero_days": len(zero_day_df),
                "detected_threats": zd_caught,
                "catch_rate": round(hybrid_zd_catch_rate, 4),
                "flagged_as_zero_day": sum(1 for v in verdicts if v.status == "SUSPECTED_ZERO_DAY"),
                "flagged_as_known_attack": sum(1 for v in verdicts if v.status == "KNOWN_ATTACK"),
            }

        return {
            "path_a_evaluation": path_a_results,
            "path_b_evaluation": path_b_results,
            "hybrid_zero_day_defense": hybrid_zd_verdicts,
        }

    def save(self, models_dir: Optional[Union[str, Path]] = None) -> None:
        """Save all pipeline components."""
        target_dir = Path(models_dir) if models_dir else self.models_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessor.save(target_dir / "preprocessor.joblib")
        self.path_a.save(target_dir)
        self.path_b.save(target_dir)
        logger.info(f"CyberGuard complete engine serialized to {target_dir}")

    def load(self, models_dir: Optional[Union[str, Path]] = None) -> None:
        """Load all pipeline components."""
        target_dir = Path(models_dir) if models_dir else self.models_dir

        prep_file = target_dir / "preprocessor.joblib"
        if not prep_file.exists():
            raise FileNotFoundError(f"Cannot find preprocessor at {prep_file}. Train model first.")

        self.preprocessor = DataPreprocessor.load(prep_file)
        self.path_a = KnownAttackClassifier()
        self.path_a.load(target_dir)
        self.path_b = ZeroDayAutoencoder()
        self.path_b.load(target_dir)
        self.is_ready = True
        logger.info(f"CyberGuard complete engine loaded from {target_dir}")


# ============================================================================
# CLI & Demonstration Driver
# ============================================================================

def run_demonstration(output_dir: Path):
    """
    Executes a complete self-contained CyberGuard demonstration:
    1. Generates 15,000 synthetic network flow events.
    2. Holds out simulated Zero-Day attacks for zero-shot testing.
    3. Trains Path A (XGBoost) and Path B (Autoencoder).
    4. Evaluates classification metrics and zero-day detection rate.
    5. Displays structured threat triage table for mixed live traffic.
    """
    print("\n" + "=" * 80)
    print("  CYBERGUARD: DUAL-PATH AI NETWORK INTRUSION DETECTION DEMONSTRATION  ")
    print("=" * 80 + "\n")

    data_dir = output_dir.parent / "data"
    synthetic_dir = data_dir / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir

    # 1. Generate Dataset
    print("[1/5] Generating Synthetic Network Traffic Dataset...")
    full_df = SyntheticTrafficGenerator.generate_dataset(
        n_samples=16000,
        benign_ratio=0.60,
        include_zero_days=True,
        zero_day_ratio=0.10,
        random_state=42,
    )

    # Separate Zero-Days to simulate unseen novel attacks
    is_zero_day = full_df["label"].isin(SyntheticTrafficGenerator.ZERO_DAY_ATTACK_TYPES)
    zero_day_df = full_df[is_zero_day].reset_index(drop=True)
    known_traffic_df = full_df[~is_zero_day].reset_index(drop=True)

    # Split known traffic into train and test
    train_df, test_df = train_test_split(
        known_traffic_df, test_size=0.25, random_state=42, stratify=known_traffic_df["label"]
    )

    # Export datasets
    train_df.to_csv(synthetic_dir / "train_known_traffic.csv", index=False)
    test_df.to_csv(synthetic_dir / "test_known_traffic.csv", index=False)
    zero_day_df.to_csv(synthetic_dir / "test_zero_day_attacks.csv", index=False)
    print(f"      - Training samples (Known & Benign) : {len(train_df):,}")
    print(f"      - Test samples (Known & Benign)     : {len(test_df):,}")
    print(f"      - Held-out Zero-Day Test samples    : {len(zero_day_df):,}")
    print(f"      - Datasets saved to: {synthetic_dir}")

    # 2. Train Engine
    print("\n[2/5] Training Dual-Path CyberGuard Models...")
    engine = CyberGuardEngine(models_dir=models_dir)
    train_summary = engine.train_pipeline(
        train_df,
        val_split=0.20,
        epochs_ae=30,
    )
    print(f"      - Path A (XGBoost) Train Accuracy   : {train_summary['path_a']['train_accuracy']*100:.2f}%")
    print(f"      - Path A Validation Accuracy        : {train_summary['path_a'].get('val_accuracy', 0)*100:.2f}%")
    print(f"      - Path B (Autoencoder) Threshold    : {train_summary['path_b']['anomaly_threshold']:.5f}")
    print(f"      - Baseline Benign Loss (Mean / Std) : {train_summary['path_b']['mean_benign_loss']:.5f} / {train_summary['path_b']['std_benign_loss']:.5f}")

    # 3. System Evaluation
    print("\n[3/5] Evaluating Models & Hybrid Defense...")
    eval_results = engine.evaluate_system(test_df=test_df, zero_day_df=zero_day_df)
    
    path_a_eval = eval_results["path_a_evaluation"]
    print(f"\n      --- Path A: Known Attack Classifier Metrics ---")
    print(f"      Overall Accuracy: {path_a_eval['accuracy'] * 100:.2f}%")
    print(f"      Weighted F1-Score: {path_a_eval['f1_weighted']:.4f}")
    print(f"      Weighted Recall:   {path_a_eval['recall_weighted']:.4f}")

    path_b_eval = eval_results["path_b_evaluation"]
    print(f"\n      --- Path B: Zero-Day Anomaly Detection Metrics ---")
    print(f"      Zero-Day Catch Rate (TPR) : {path_b_eval['zero_day_detection_rate_tpr'] * 100:.2f}%")
    print(f"      Benign False Positive Rate : {path_b_eval['false_positive_rate_fpr'] * 100:.2f}%")
    print(f"      Zero-Day Anomaly AUC-ROC   : {path_b_eval['auc_roc']:.4f}")

    hybrid_eval = eval_results["hybrid_zero_day_defense"]
    print(f"\n      --- Hybrid Engine Zero-Day Triage Performance ---")
    print(f"      Unseen Zero-Day Exploit Detection Rate: {hybrid_eval['catch_rate'] * 100:.2f}%")
    print(f"      Correctly Triaged as Suspicious/Zero-Day: {hybrid_eval['detected_threats']} / {hybrid_eval['total_zero_days']}")

    # 4. Live Stream Simulation
    print("\n[4/5] Simulating Live Traffic Stream & Threat Intelligence Triage...")
    # Mix benign, known attacks, and zero-day samples
    live_sample_df = pd.concat([
        test_df[test_df["label"] == "BENIGN"].head(4),
        test_df[test_df["label"] == "DoS_SYN_Flood"].head(2),
        test_df[test_df["label"] == "Port_Scan"].head(2),
        test_df[test_df["label"] == "Brute_Force_SSH"].head(1),
        zero_day_df.head(3),
    ]).sample(frac=1.0, random_state=99).reset_index(drop=True)

    verdicts = engine.analyze_traffic(live_sample_df)

    print("\n" + "-" * 115)
    print(f"{'#':<3} | {'GROUND TRUTH':<22} | {'HYBRID STATUS':<18} | {'SEVERITY':<8} | {'PATH A CLASS (CONF)':<24} | {'PATH B MSE':<10} | {'ANOMALY':<7}")
    print("-" * 115)
    for i, (idx_row, v) in enumerate(zip(live_sample_df.iterrows(), verdicts)):
        actual_label = idx_row[1]["label"]
        p_a_str = f"{v.path_a_prediction} ({v.path_a_confidence*100:.0f}%)"
        print(
            f"{i+1:<3} | {actual_label:<22} | {v.status:<18} | {v.threat_level:<8} | "
            f"{p_a_str:<24} | {v.path_b_recon_error:<10.5f} | {str(v.path_b_is_anomaly):<7}"
        )
    print("-" * 115)

    # 5. Feature Importances
    print("\n[5/5] Top Network Flow Features (Path A XGBoost):")
    top_features = engine.path_a.get_feature_importances(engine.preprocessor.feature_columns)[:8]
    for rank, f_info in enumerate(top_features, 1):
        print(f"      {rank}. {f_info['feature']:<30} (Score: {f_info['importance']:.4f})")

    print("\n" + "=" * 80)
    print("  CYBERGUARD DEMONSTRATION COMPLETE - ARTIFACTS READY IN models/ AND data/  ")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="CyberGuard AI Network Intrusion Detection Engine")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["demo", "train", "evaluate", "predict"],
        default="demo",
        help="Operation mode (default: demo)",
    )
    parser.add_argument("--data", type=str, default=None, help="Path to dataset CSV file")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory for saved model artifacts")
    parser.add_argument("--output", type=str, default=None, help="Path to save prediction output (CSV/JSON)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for Deep Autoencoder")

    args = parser.parse_args()
    models_dir = Path(args.models_dir)

    if args.mode == "demo":
        run_demonstration(models_dir)
    elif args.mode == "train":
        if not args.data:
            print("Error: --data <path_to_csv> is required for training mode.")
            sys.exit(1)
        df = pd.read_csv(args.data)
        engine = CyberGuardEngine(models_dir=models_dir)
        engine.train_pipeline(df, epochs_ae=args.epochs)
    elif args.mode == "evaluate":
        if not args.data:
            print("Error: --data <path_to_csv> is required for evaluate mode.")
            sys.exit(1)
        df = pd.read_csv(args.data)
        engine = CyberGuardEngine(models_dir=models_dir)
        engine.load()
        results = engine.evaluate_system(df)
        print(json.dumps(results, indent=2))
    elif args.mode == "predict":
        if not args.data:
            print("Error: --data <path_to_csv> is required for predict mode.")
            sys.exit(1)
        df = pd.read_csv(args.data)
        engine = CyberGuardEngine(models_dir=models_dir)
        engine.load()
        verdicts = engine.analyze_traffic(df)
        verdict_dicts = [v.to_dict() for v in verdicts]
        if args.output:
            out_p = Path(args.output)
            if out_p.suffix == ".json":
                with open(out_p, "w") as f:
                    json.dump(verdict_dicts, f, indent=2)
            else:
                pd.DataFrame(verdict_dicts).to_csv(out_p, index=False)
            print(f"Predictions saved to {args.output}")
        else:
            print(json.dumps(verdict_dicts[:10], indent=2))


if __name__ == "__main__":
    main()
