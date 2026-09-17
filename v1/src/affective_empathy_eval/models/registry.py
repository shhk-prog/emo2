from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import yaml


@dataclass
class ModelSpec:
    model_id: str
    format: str  # "plain" or "chat"


@dataclass
class ModelFamilyConfig:
    family_id: str
    family_name: str
    scale: str
    num_layers: int
    hidden_dim: int
    architecture: str
    role: str
    base_model: ModelSpec
    instruct_model: ModelSpec

    def get_model_spec(self, alignment: str) -> ModelSpec:
        alignment_lower = alignment.lower()
        if alignment_lower == "base":
            return self.base_model
        elif alignment_lower in ("instruct", "it"):
            return self.instruct_model
        else:
            raise ValueError(f"Unknown alignment: {alignment}. Expected 'base' or 'instruct'.")


class ModelRegistry:
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            # デフォルトでプロジェクトルートの configs/models.yaml を探す
            current_dir = Path(__file__).resolve().parent
            # emo/src/affective_empathy_eval/models -> emo
            project_root = current_dir.parents[3]
            config_path = project_root / "configs" / "models.yaml"

        self.config_path = Path(config_path)
        self.families: Dict[str, ModelFamilyConfig] = {}
        self._load()

    def _load(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Model config not found at {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        families_data = data.get("families", {})
        for fid, finfo in families_data.items():
            models = finfo.get("models", {})
            base_spec = ModelSpec(
                model_id=models["base"]["model_id"],
                format=models["base"]["format"],
            )
            instruct_spec = ModelSpec(
                model_id=models["instruct"]["model_id"],
                format=models["instruct"]["format"],
            )
            self.families[fid] = ModelFamilyConfig(
                family_id=fid,
                family_name=finfo.get("family_name", fid),
                scale=finfo.get("scale", ""),
                num_layers=int(finfo["num_layers"]),
                hidden_dim=int(finfo["hidden_dim"]),
                architecture=finfo.get("architecture", ""),
                role=finfo.get("role", "replication"),
                base_model=base_spec,
                instruct_model=instruct_spec,
            )

    def get_family(self, family_id: str) -> ModelFamilyConfig:
        if family_id not in self.families:
            raise KeyError(f"Family {family_id} not found. Available: {list(self.families.keys())}")
        return self.families[family_id]

    def list_families(self) -> List[str]:
        return list(self.families.keys())

    def get_relative_depth(self, family_id: str, layer: int) -> float:
        num_layers = self.get_family(family_id).num_layers
        if num_layers <= 1:
            return 0.0
        return layer / (num_layers - 1)


_GLOBAL_REGISTRY: Optional[ModelRegistry] = None


def get_registry(config_path: Optional[Path] = None) -> ModelRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None or config_path is not None:
        _GLOBAL_REGISTRY = ModelRegistry(config_path)
    return _GLOBAL_REGISTRY
