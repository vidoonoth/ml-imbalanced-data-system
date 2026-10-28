"""Inference service for the SCARF normal/malware classifier."""

import json
import re
from collections import UserList
from datetime import date, datetime
from ipaddress import ip_address
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

ML_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ML_DIR / "model_artifacts"

DAY_MAP = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

# The training CSV uses src/dst names. These aliases keep the API compatible
# with the Laravel schema and with common firewall export names.
COLUMN_ALIASES = {
    "source_ip": "src_ip",
    "srcip": "src_ip",
    "destination_ip": "dst_ip",
    "dest_ip": "dst_ip",
    "dstip": "dst_ip",
    "source_port": "src_port",
    "srcport": "src_port",
    "destination_port": "dst_port",
    "dest_port": "dst_port",
    "dstport": "dst_port",
    "update_time": "timestamp",
    "geo_src": "src_country",
    "geo_dst": "dst_country",
    "policy": "policy_name",
    "disposition": "application",
}

LOG_FIELD_ALIASES = {
    "disp": "action",
    "pri": "severity",
    "src_intf": "source_interface",
    "dst_intf": "destination_interface",
    "geo_src": "src_country",
    "geo_dst": "dst_country",
    "policy": "policy_name",
}

LOG_PAIR_PATTERN = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)=(.*?)(?=,\s*[A-Za-z_][A-Za-z0-9_]*=|$)"
)


class SCARFEncoder(nn.Module):
    """Encoder architecture used by the new notebook checkpoint."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, embedding_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SCARFClassifier(nn.Module):
    """Classifier architecture saved in scarf_classifier.pt."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        encoder_hidden_dim: int = 256,
        classifier_hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = SCARFEncoder(input_dim, encoder_hidden_dim, embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Linear(classifier_hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class DetectionService:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        self.artifacts_dir = Path(artifacts_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_payload: dict[str, Any] | None = None
        self.preprocessor = None
        self.model: SCARFClassifier | None = None
        # Kept as an attribute for callers that previously inspected encoder.
        self.encoder: SCARFEncoder | None = None
        self.classifier: SCARFClassifier | None = None
        self.decision_threshold = 0.5
        self._load_models()

    def _load_models(self) -> None:
        """Load metadata, fitted preprocessing, and the new full classifier."""
        self._patch_sklearn_compatibility()

        metadata_path = self.artifacts_dir / "metadata.json"
        preprocessor_path = self.artifacts_dir / "preprocessor.joblib"
        classifier_path = self.artifacts_dir / "scarf_classifier.pt"

        missing = [
            str(path)
            for path in (metadata_path, preprocessor_path, classifier_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Model artifacts tidak lengkap; file tidak ditemukan: " + ", ".join(missing)
            )

        with metadata_path.open(encoding="utf-8") as file:
            self.feature_payload = json.load(file)

        self.preprocessor = joblib.load(preprocessor_path)
        classifier_data = torch.load(
            classifier_path,
            map_location=self.device,
            weights_only=False,
        )

        input_dim = int(
            classifier_data.get("input_dim", self.feature_payload.get("input_dim", 0))
        )
        embedding_dim = int(
            classifier_data.get("embedding_dim", self.feature_payload.get("embedding_dim", 64))
        )
        classifier_hidden_dim = int(classifier_data.get("hidden_dim", 256))
        # The checkpoint does not store this separately. It is the notebook's
        # SCARF_HIDDEN_DIM and can also be inferred from the first weight.
        encoder_hidden_dim = 256

        self.model = SCARFClassifier(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            classifier_hidden_dim=classifier_hidden_dim,
        ).to(self.device)
        self.model.load_state_dict(classifier_data["model_state_dict"])
        self.model.eval()
        self.encoder = self.model.encoder
        self.classifier = self.model
        self.decision_threshold = float(
            self.feature_payload.get("decision_threshold", 0.5)
        )

    def _patch_sklearn_compatibility(self) -> None:
        """Support loading ColumnTransformer artifacts across sklearn versions."""
        try:
            import sklearn.compose._column_transformer as column_transformer

            if hasattr(column_transformer, "_RemainderColsList"):
                return

            class _RemainderColsList(UserList):
                def __init__(
                    self,
                    cols,
                    *,
                    future_dtype=None,
                    warning_was_emitted=False,
                    warning_enabled=True,
                ):
                    super().__init__(cols)
                    self.future_dtype = future_dtype
                    self.warning_was_emitted = warning_was_emitted
                    self.warning_enabled = warning_enabled

            column_transformer._RemainderColsList = _RemainderColsList
        except ImportError:
            pass

    def is_ready(self) -> bool:
        return all(
            value is not None
            for value in (self.feature_payload, self.preprocessor, self.model)
        )

    def detect_single(self, record: dict) -> dict:
        results = self.detect_batch([record])
        return results[0] if results else {}

    def detect_batch(self, records: list[dict]) -> list[dict]:
        if not records:
            return []

        frame = self._normalize_columns(pd.DataFrame(records))
        feature_columns = list(self.feature_payload.get("feature_columns", []))
        numeric_features = set(self.feature_payload.get("numeric_features", []))

        # Match predict_new_dataframe() in the notebook while also allowing a
        # caller to send already-engineered features.
        if not set(feature_columns).issubset(frame.columns):
            frame = self._prepare_features(frame)

        # Production records can omit optional source columns. The fitted
        # imputer handles missing numeric values; unknown is ignored by the
        # fitted OneHotEncoder for categorical values.
        for column in feature_columns:
            if column not in frame.columns:
                frame[column] = np.nan if column in numeric_features else "unknown"

        X = self.preprocessor.transform(frame[feature_columns])
        X = self._to_dense_float32(X)
        embeddings = self._generate_embeddings(X)
        predictions, probabilities = self._predict(embeddings)

        return self._build_results(frame, predictions, probabilities)

    def _normalize_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        renamed: dict[Any, str] = {}
        seen: dict[str, int] = {}

        for column in frame.columns:
            normalized = re.sub(
                r"[^a-z0-9_]+", "_", str(column).strip().lower()
            ).strip("_")
            if normalized in seen:
                seen[normalized] += 1
                normalized = f"{normalized}_{seen[normalized]}"
            else:
                seen[normalized] = 1
            renamed[column] = normalized

        frame = frame.rename(columns=renamed)
        frame = self._expand_log_fields(frame)

        for source, target in COLUMN_ALIASES.items():
            if source in frame.columns and target not in frame.columns:
                frame[target] = frame[source]

        return frame

    def _expand_log_fields(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Extract comma-separated key=value fields from a raw log column."""
        if "log" not in frame.columns:
            return frame

        for index, value in frame["log"].items():
            if pd.isna(value):
                continue

            parsed = self._parse_log_pairs(str(value))
            if not parsed:
                continue

            for key, parsed_value in parsed.items():
                target = LOG_FIELD_ALIASES.get(key, key)
                if target not in frame.columns:
                    frame[target] = np.nan
                if pd.isna(frame.at[index, target]) or frame.at[index, target] == "":
                    frame.at[index, target] = parsed_value

            if "event_type" not in frame.columns:
                frame["event_type"] = np.nan
            if pd.isna(frame.at[index, "event_type"]) or frame.at[index, "event_type"] == "":
                frame.at[index, "event_type"] = str(value).split(",", 1)[0].strip()

        return frame

    @staticmethod
    def _parse_log_pairs(value: str) -> dict[str, str]:
        return {
            key.strip().lower(): raw_value.strip().strip('"')
            for key, raw_value in LOG_PAIR_PATTERN.findall(value)
        }

    def _prepare_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Mirror engineer_features() from the training notebook."""
        frame = frame.copy()
        for column in [
            "timestamp",
            "hour",
            "day_of_week",
            "protocol",
            "src_ip",
            "dst_ip",
            "message",
        ]:
            if column not in frame.columns:
                frame[column] = np.nan

        timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
        hour = (
            pd.to_numeric(frame["hour"], errors="coerce")
            .fillna(timestamp.dt.hour)
            .fillna(0.0)
        )
        minute = timestamp.dt.minute.fillna(0.0)
        second = timestamp.dt.second.fillna(0.0)
        day = (
            frame["day_of_week"]
            .astype("string")
            .str.strip()
            .str.lower()
            .map(DAY_MAP)
            .fillna(timestamp.dt.dayofweek)
            .fillna(0)
            .astype(float)
        )

        frame["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        frame["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        frame["minute_sin"] = np.sin(2 * np.pi * minute / 60.0)
        frame["minute_cos"] = np.cos(2 * np.pi * minute / 60.0)
        frame["second_sin"] = np.sin(2 * np.pi * second / 60.0)
        frame["second_cos"] = np.cos(2 * np.pi * second / 60.0)
        frame["day_sin"] = np.sin(2 * np.pi * day / 7.0)
        frame["day_cos"] = np.cos(2 * np.pi * day / 7.0)

        for column in ["src_port", "dst_port", "severity"]:
            if column not in frame.columns:
                frame[column] = np.nan
        frame["src_port"] = pd.to_numeric(frame["src_port"], errors="coerce")
        frame["dst_port"] = pd.to_numeric(frame["dst_port"], errors="coerce")
        frame["severity_num"] = pd.to_numeric(frame["severity"], errors="coerce")

        protocol = frame["protocol"].fillna("unknown").astype(str).str.lower()
        parts = protocol.str.split("/", n=1, expand=True)
        frame["protocol_service"] = parts[0].replace("", "unknown").fillna("unknown")
        frame["protocol_transport"] = (
            parts[1].replace("", "unknown").fillna("unknown")
            if parts.shape[1] > 1
            else "unknown"
        )

        source_features = pd.DataFrame(
            frame["src_ip"].apply(lambda value: self._ip_features(value, "src")).tolist(),
            index=frame.index,
        )
        destination_features = pd.DataFrame(
            frame["dst_ip"].apply(lambda value: self._ip_features(value, "dst")).tolist(),
            index=frame.index,
        )
        frame = pd.concat([frame, source_features, destination_features], axis=1)

        message = frame["message"].fillna("unknown").astype(str)
        prefix = (
            message.str.rsplit(")", n=1)
            .str[-1]
            .str.split(":", n=1)
            .str[0]
            .str.strip()
            .str.lower()
        )
        frame["message_prefix"] = prefix.replace("", "unknown").fillna("unknown")

        return frame

    @staticmethod
    def _ip_features(value: Any, prefix: str) -> dict[str, int]:
        result = {
            f"{prefix}_ip_is_private": 0,
            f"{prefix}_ip_is_global": 0,
            f"{prefix}_ip_version": 0,
            f"{prefix}_ip_octet1": 0,
            f"{prefix}_ip_octet2": 0,
        }
        try:
            ip_object = ip_address(str(value).strip())
        except (TypeError, ValueError):
            return result

        result[f"{prefix}_ip_is_private"] = int(ip_object.is_private)
        result[f"{prefix}_ip_is_global"] = int(ip_object.is_global)
        result[f"{prefix}_ip_version"] = int(ip_object.version)
        if ip_object.version == 4:
            octets = str(ip_object).split(".")
            result[f"{prefix}_ip_octet1"] = int(octets[0])
            result[f"{prefix}_ip_octet2"] = int(octets[1])
        return result

    @staticmethod
    def _to_dense_float32(matrix: Any) -> np.ndarray:
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        return np.asarray(matrix, dtype=np.float32)

    def _generate_embeddings(self, X: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            embeddings = self.encoder(tensor).cpu().numpy()
        return embeddings.astype(np.float32)

    def _predict(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            # The checkpoint contains encoder + head. Embeddings were already
            # generated above, so apply only the classifier head here.
            logits = self.classifier.head(tensor)
            probabilities = F.softmax(logits, dim=1).cpu().numpy()
            predictions = (
                probabilities[:, 1] >= self.decision_threshold
            ).astype(np.int64)
        return predictions, probabilities.astype(np.float32)

    def _build_results(
        self,
        frame: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> list[dict]:
        label_map = {0: "Normal", 1: "Malware"}
        results = []

        optional_fields = [
            "update_time",
            "sn",
            "log",
            "log_type",
            "event_type",
            "event_name",
            "action",
            "severity",
            "priority",
            "message",
            "src_country",
            "dst_country",
            "geo_src",
            "geo_dst",
            "policy_name",
            "policy",
            "application",
            "disposition",
            "source_interface",
            "destination_interface",
            "pckt_len",
            "ttl",
            "sent_bytes",
            "rcvd_bytes",
            "url",
            "url_category",
            "date",
            "is_blocked",
            "is_allowed",
        ]

        for index, (_, row) in enumerate(frame.iterrows()):
            prediction = int(predictions[index])
            probability_normal = float(probabilities[index, 0])
            probability_attack = float(probabilities[index, 1])
            result = {
                "prediction": prediction,
                "prediction_label": label_map.get(prediction, str(prediction)),
                "confidence": max(probability_normal, probability_attack),
                "probability_normal": probability_normal,
                "probability_attack": probability_attack,
                "malware_probability": probability_attack,
                "source_ip": self._string_value(row.get("src_ip")),
                "destination_ip": self._string_value(row.get("dst_ip")),
                "source_port": self._int_value(row.get("src_port")),
                "destination_port": self._int_value(row.get("dst_port")),
                "protocol": self._string_value(row.get("protocol")),
                "timestamp": self._datetime_value(row.get("timestamp")),
            }

            for field in ("_dataset_id", "_row_number"):
                value = row.get(field)
                if pd.notna(value):
                    result[field] = int(value)

            for field in optional_fields:
                if field not in row.index:
                    continue
                value = self._json_value(row.get(field))
                if value is not None:
                    result[field] = value

            results.append(result)

        return results

    @staticmethod
    def _string_value(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        return str(value)

    @staticmethod
    def _int_value(value: Any) -> int | None:
        if value is None or pd.isna(value):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _datetime_value(cls, value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            return parsed.isoformat()
        return str(value)

    @classmethod
    def _json_value(cls, value: Any) -> Any:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (pd.Timestamp, datetime, date)):
            return value.isoformat()
        return value if isinstance(value, (str, int, float, bool)) else str(value)
