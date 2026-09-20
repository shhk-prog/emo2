import json
import os
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Optional

from dataclasses import dataclass, asdict

@dataclass
class ExtractionManifest:
    """
    Manifest for a single extracted activation tensor.
    Keeps track of all necessary metadata to ensure reproducibility and correct alignment.
    """
    tensor_path: str                 # Relative path to the saved .npy or .npz file
    stimulus_id: str                 # Unique ID of the stimulus
    condition: str                   # Extraction condition (e.g., 'empty_baseline', 'post_reported_va')
    model_name: str                  # Name of the model
    model_revision: str              # Model revision / commit hash
    layer: int                       # Layer number from which the activation was extracted
    extraction_position: str         # The position identifier (e.g., 'stimulus_last_token')
    token_offset: Optional[int]      # Actual absolute index in the sequence (if applicable)
    prompt_hash: str                 # Hash of the exact prompt used
    tensor_shape: List[int]          # Shape of the extracted tensor
    dtype: str                       # Data type (e.g., 'float16')

class ManifestManager:
    """
    Manages the saving and loading of extraction manifests.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.manifests: List[ExtractionManifest] = []
        os.makedirs(self.output_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.output_dir, "activation_manifest.jsonl")

    def add_manifest(self, manifest: ExtractionManifest):
        self.manifests.append(manifest)
        
    def save(self):
        """Append all currently held manifests to the JSONL file and clear the buffer."""
        if not self.manifests:
            return
            
        with open(self.manifest_path, "a") as f:
            for manifest in self.manifests:
                f.write(json.dumps(asdict(manifest)) + "\n")
        
        self.manifests = []
        
    def load_all_as_dataframe(self) -> pd.DataFrame:
        """Loads the entire manifest file into a pandas DataFrame."""
        if not os.path.exists(self.manifest_path):
            return pd.DataFrame()
            
        records = []
        with open(self.manifest_path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return pd.DataFrame(records)


import hashlib


DEFAULT_INTERVENTION_VERSION = "none"
DEFAULT_CODE_VERSION = "2.2.0"
DEFAULT_PROMPT_VERSION = "1.0.0"
DEFAULT_CANDIDATE_SPACE = "VA_81"


def compute_file_hash(path: str | Path) -> str:
    """AGENTS.md 1.2 / Item 11: ファイルの実内容に基づく SHA256 ハッシュを算出"""
    p = Path(path)
    if not p.is_file():
        return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def compute_string_or_dict_hash(obj: Any) -> str:
    """オブジェクト（辞書、文字列、パス等）の SHA256 ハッシュを算出。パスの場合はファイル内容から算出。"""
    if isinstance(obj, dict):
        s = json.dumps(obj, sort_keys=True)
    elif isinstance(obj, (str, Path)):
        if os.path.isfile(str(obj)):
            return compute_file_hash(obj)
        s = str(obj)
    else:
        s = str(obj)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def generate_run_id(git_sha: Optional[str] = None, config_hash: Optional[str] = None) -> str:
    """
    AGENTS.md 5.3 / Item 25, 26 準拠の衝突防止一意 run_id 生成関数:
    YYYYMMDDTHHMMSSffffffZ_<git-short-sha>_<config-short-hash>_<uuid8>
    """
    from datetime import datetime, timezone
    import subprocess
    import uuid

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if git_sha is None:
        try:
            git_sha = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
        except Exception:
            git_sha = "unknown"
    if config_hash is None:
        config_hash = "00000000"
    uid = uuid.uuid4().hex[:8]
    return f"{timestamp}_{git_sha}_{config_hash[:8]}_{uid}"


@dataclass
class RunManifest:
    """Run manifest storing metadata for reproducibility and cache validation."""
    run_type: str
    model_name: str
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp_utc: str
    git_commit: str
    run_id: str = "unknown"
    config_hash: str = "unknown"
    dataset_hash: str = "unknown"
    model_revision: str = "main"
    tokenizer_revision: str = "main"
    prompt_version: str = DEFAULT_PROMPT_VERSION
    prompt_hash: str = "unknown"
    candidate_space: str = DEFAULT_CANDIDATE_SPACE
    measurement_space: str = "VA_81"
    actual_dtype: str = "bfloat16"
    template_mode: str = "system_user"
    seed: int = 42
    code_version: str = DEFAULT_CODE_VERSION
    intervention_version: str = DEFAULT_INTERVENTION_VERSION
    sequence_likelihood_normalization: str = "token_mean"
    temperature: float = 1.0
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_prompt_hash(prompt_text_or_template: str) -> str:
    """AGENTS.md 5.2 準拠: sha256(template.encode()) の先頭16文字"""
    return hashlib.sha256(prompt_text_or_template.encode("utf-8")).hexdigest()[:16]


def create_run_manifest(
    run_type: str,
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    dataset_path: Optional[str] = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    prompt_hash: Optional[str] = None,
    candidate_space: str = DEFAULT_CANDIDATE_SPACE,
    measurement_space: str = "VA_81",
    actual_dtype: str = "bfloat16",
    template_mode: str = "system_user",
    seed: int = 42,
    intervention_version: str = DEFAULT_INTERVENTION_VERSION,
    model_revision: str = "main",
    tokenizer_revision: str = "main",
    run_id: Optional[str] = None,
    dry_run: bool = False,
    dataset_hash: Optional[str] = None,
    sequence_likelihood_normalization: Optional[str] = None,
    temperature: Optional[float] = None,
) -> RunManifest:
    from datetime import datetime, timezone
    import subprocess

    try:
        git_sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_sha = "unknown"

    cfg = config or {}
    meta = metadata or {}

    if not dataset_path and "dataset_path" in cfg:
        dataset_path = str(cfg["dataset_path"])
    elif not dataset_path and isinstance(cfg.get("dataset"), dict) and "path" in cfg["dataset"]:
        dataset_path = str(cfg["dataset"]["path"])

    if seed == 42 and "seed" in cfg:
        try:
            seed = int(cfg["seed"])
        except Exception:
            pass

    if model_revision == "main" and "model_revision" in cfg:
        model_revision = str(cfg["model_revision"])
    if tokenizer_revision == "main" and "tokenizer_revision" in cfg:
        tokenizer_revision = str(cfg["tokenizer_revision"])
    if candidate_space == DEFAULT_CANDIDATE_SPACE and "candidate_space" in cfg:
        candidate_space = str(cfg["candidate_space"])
    if measurement_space == "VA_81" and "measurement_space" in cfg:
        measurement_space = str(cfg["measurement_space"])
    if actual_dtype == "bfloat16" and "actual_dtype" in cfg:
        actual_dtype = str(cfg["actual_dtype"])
    if template_mode == "system_user" and "template_mode" in cfg:
        template_mode = str(cfg["template_mode"])

    seq_norm = sequence_likelihood_normalization
    if seq_norm is None:
        if "sequence_likelihood_normalization" in cfg:
            seq_norm = str(cfg["sequence_likelihood_normalization"])
        elif isinstance(cfg.get("sequence_likelihood"), dict) and cfg["sequence_likelihood"].get("normalize_length", True):
            seq_norm = "token_mean"
        elif isinstance(cfg.get("sequence_likelihood"), dict) and not cfg["sequence_likelihood"].get("normalize_length", True):
            seq_norm = "raw_sum"
        else:
            seq_norm = "token_mean"

    temp = temperature
    if temp is None:
        if "temperature" in cfg:
            temp = float(cfg["temperature"])
        elif isinstance(cfg.get("sequence_likelihood"), dict) and "temperature" in cfg["sequence_likelihood"]:
            temp = float(cfg["sequence_likelihood"]["temperature"])
        else:
            temp = 1.0

    cfg_hash = compute_string_or_dict_hash(cfg)
    if dataset_hash is not None:
        ds_hash = dataset_hash
    elif dataset_path and Path(dataset_path).exists():
        ds_hash = compute_file_hash(Path(dataset_path))
    elif dataset_path:
        ds_hash = compute_string_or_dict_hash(dataset_path)
    else:
        ds_hash = "unknown"

    if run_id is None:
        run_id = generate_run_id(git_sha=git_sha, config_hash=cfg_hash)

    if prompt_hash is None:
        if "prompt_hash" in cfg:
            prompt_hash = str(cfg["prompt_hash"])
        elif "prompt_template" in cfg:
            prompt_hash = compute_prompt_hash(str(cfg["prompt_template"]))
        else:
            prompt_hash = "unknown"

    return RunManifest(
        run_type=run_type,
        model_name=model_name,
        config=cfg,
        metadata=meta,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=git_sha,
        run_id=run_id,
        config_hash=cfg_hash,
        dataset_hash=ds_hash,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        prompt_version=prompt_version,
        prompt_hash=prompt_hash,
        candidate_space=candidate_space,
        measurement_space=measurement_space,
        actual_dtype=actual_dtype,
        template_mode=template_mode,
        seed=seed,
        code_version=DEFAULT_CODE_VERSION,
        intervention_version=intervention_version,
        sequence_likelihood_normalization=seq_norm,
        temperature=temp,
        dry_run=dry_run,
    )


def is_manifest_matching(
    manifest_path: str,
    expected_model_name: Optional[str] = None,
    expected_intervention_version: Optional[str] = None,
    expected_candidate_space: Optional[str] = None,
    expected_measurement_space: Optional[str] = None,
    expected_prompt_version: Optional[str] = None,
    expected_prompt_hash: Optional[str] = None,
    expected_config_hash: Optional[str] = None,
    expected_dataset_hash: Optional[str] = None,
    expected_code_version: Optional[str] = DEFAULT_CODE_VERSION,
    expected_model_revision: Optional[str] = None,
    expected_tokenizer_revision: Optional[str] = None,
    expected_git_commit: Optional[str] = None,
    expected_dry_run: Optional[bool] = None,
    expected_sequence_likelihood_normalization: Optional[str] = None,
) -> bool:
    """
    キャッシュの有効性を検証する。
    旧設定・旧コード・旧データセット・dry-run結果のキャッシュと新パイプライン成果物の混在を防止。
    """
    if not os.path.exists(manifest_path):
        return False

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if expected_dry_run is not None and data.get("dry_run", False) != expected_dry_run:
            return False

        if expected_model_name and data.get("model_name") != expected_model_name:
            return False

        if expected_intervention_version and data.get("intervention_version") != expected_intervention_version:
            return False

        if expected_candidate_space and data.get("candidate_space") != expected_candidate_space:
            return False

        if expected_measurement_space and data.get("measurement_space") != expected_measurement_space:
            return False

        if expected_prompt_version and data.get("prompt_version") != expected_prompt_version:
            return False

        if expected_prompt_hash and data.get("prompt_hash") != expected_prompt_hash:
            return False

        if expected_config_hash and data.get("config_hash") != expected_config_hash:
            return False

        if expected_dataset_hash and data.get("dataset_hash") != expected_dataset_hash:
            return False

        if expected_code_version and data.get("code_version") != expected_code_version:
            return False

        if expected_model_revision and data.get("model_revision") != expected_model_revision:
            return False

        if expected_tokenizer_revision and data.get("tokenizer_revision") != expected_tokenizer_revision:
            return False

        if expected_git_commit and data.get("git_commit") != expected_git_commit:
            return False

        if expected_sequence_likelihood_normalization and data.get("sequence_likelihood_normalization") != expected_sequence_likelihood_normalization:
            return False

        return True
    except Exception:
        return False
