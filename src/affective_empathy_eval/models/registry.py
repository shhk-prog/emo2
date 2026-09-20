from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

logger = logging.getLogger(__name__)

# オフライン実行時・テスト時・HFアクセス不可時のフォールバック既知次元テーブル
# (num_layers, hidden_dim)
KNOWN_MODEL_DIMS: Dict[str, Tuple[int, int]] = {
    # Qwen 2.5
    "Qwen/Qwen2.5-1.5B": (28, 1536),
    "Qwen/Qwen2.5-1.5B-Instruct": (28, 1536),
    "Qwen/Qwen2.5-0.5B": (24, 896),
    "Qwen/Qwen2.5-0.5B-Instruct": (24, 896),
    "Qwen/Qwen2.5-7B": (28, 3584),
    "Qwen/Qwen2.5-7B-Instruct": (28, 3584),
    # Llama 3.2
    "meta-llama/Llama-3.2-1B": (16, 2048),
    "meta-llama/Llama-3.2-1B-Instruct": (16, 2048),
    "meta-llama/Llama-3.2-3B": (28, 3072),
    "meta-llama/Llama-3.2-3B-Instruct": (28, 3072),
    # Gemma 3 & Gemma 2
    "google/gemma-3-1b-pt": (26, 1152),
    "google/gemma-3-1b-it": (26, 1152),
    "google/gemma-2-2b": (26, 2304),
    "google/gemma-2-2b-it": (26, 2304),
    # OLMo 2
    "allenai/OLMo-2-0425-1B": (16, 2048),
    "allenai/OLMo-2-0425-1B-Instruct": (16, 2048),
    # Mistral
    "mistralai/Mistral-7B-v0.1": (32, 4096),
    "mistralai/Mistral-7B-Instruct-v0.2": (32, 4096),
    "mistralai/Mistral-7B-v0.3": (32, 4096),
    "mistralai/Mistral-7B-Instruct-v0.3": (32, 4096),
}

_DIMS_CACHE: Dict[str, Tuple[int, int]] = {}


def resolve_architecture_dims(model_id: str) -> Tuple[int, int]:
    """
    モデルIDからトランスフォーマー層数 (num_layers) と隠れ層次元 (hidden_dim) を自動取得する。
    HuggingFaceのAutoConfigから取得を試み、失敗した場合はローカルキャッシュまたは既知のフォールバックテーブルを参照する。
    """
    if model_id in _DIMS_CACHE:
        return _DIMS_CACHE[model_id]

    # 1. HuggingFace AutoConfig から動的取得を試みる
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # 層数属性の探索
        num_layers = None
        for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
            if hasattr(config, attr):
                num_layers = int(getattr(config, attr))
                break

        # 隠れ層次元属性の探索
        hidden_dim = None
        for attr in ("hidden_size", "d_model", "dim", "hidden_dim"):
            if hasattr(config, attr):
                hidden_dim = int(getattr(config, attr))
                break

        if num_layers is not None and hidden_dim is not None:
            _DIMS_CACHE[model_id] = (num_layers, hidden_dim)
            return (num_layers, hidden_dim)
    except Exception as e:
        logger.debug(f"Could not load AutoConfig for {model_id} dynamically: {e}")

    # 2. 既知のモデルフォールバック辞書を参照
    if model_id in KNOWN_MODEL_DIMS:
        dims = KNOWN_MODEL_DIMS[model_id]
        _DIMS_CACHE[model_id] = dims
        return dims

    # 3. プレフィックス一致でのフォールバック
    for known_id, dims in KNOWN_MODEL_DIMS.items():
        if known_id.lower() in model_id.lower() or model_id.lower() in known_id.lower():
            _DIMS_CACHE[model_id] = dims
            return dims

    # 4. 未知モデルの場合は安全のため例外を送出
    raise ValueError(
        f"Unable to resolve architecture dimensions for model '{model_id}'. "
        f"Please register it in KNOWN_MODEL_DIMS or ensure HuggingFace config is accessible."
    )


@dataclass
class ModelSpec:
    model_id: str
    format: str = "plain"  # "plain" or "chat"
    revision: Optional[str] = None


class ModelFamilyConfig:
    def __init__(
        self,
        family_id: str,
        family_name: str,
        scale: str,
        base_model: ModelSpec,
        instruct_model: ModelSpec,
        adapter: str,
        role: str = "replication",
        enabled: bool = True,
        num_layers: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        architecture: Optional[str] = None,
    ):
        self.family_id = family_id
        self.family_name = family_name
        self.scale = scale
        self.base_model = base_model
        self.instruct_model = instruct_model
        self.adapter = adapter
        self.role = role
        self.enabled = enabled
        self.architecture = architecture or adapter
        self._num_layers = num_layers
        self._hidden_dim = hidden_dim

    def _ensure_dims(self):
        if self._num_layers is None or self._hidden_dim is None:
            # instruct または base のいずれかから次元を解決
            target_id = self.instruct_model.model_id or self.base_model.model_id
            layers, dim = resolve_architecture_dims(target_id)
            if self._num_layers is None:
                self._num_layers = layers
            if self._hidden_dim is None:
                self._hidden_dim = dim

    @property
    def num_layers(self) -> int:
        self._ensure_dims()
        return self._num_layers  # type: ignore

    @num_layers.setter
    def num_layers(self, val: int):
        self._num_layers = val

    @property
    def hidden_dim(self) -> int:
        self._ensure_dims()
        return self._hidden_dim  # type: ignore

    @hidden_dim.setter
    def hidden_dim(self, val: int):
        self._hidden_dim = val

    def get_model_spec(self, alignment: str) -> ModelSpec:
        alignment_lower = alignment.lower()
        if alignment_lower == "base":
            return self.base_model
        elif alignment_lower in ("instruct", "it"):
            return self.instruct_model
        else:
            raise ValueError(f"Unknown alignment: {alignment}. Expected 'base' or 'instruct'.")


def load_model_set(
    config_path: Optional[Path] = None,
    model_set: str = "primary_small",
) -> Dict[str, ModelFamilyConfig]:
    """
    指定された model_set (例: 'primary_small', 'scale_validation') の全モデル定義をロードする。
    """
    if config_path is None:
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parents[2]
        config_path = project_root / "configs" / "models.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    families: Dict[str, ModelFamilyConfig] = {}

    # 1. 新規構造 model_sets からのロード
    if "model_sets" in data:
        sets = data["model_sets"]
        if model_set not in sets:
            raise KeyError(f"model_set '{model_set}' not found in {config_path}. Available: {list(sets.keys())}")

        set_cfg = sets[model_set]
        for key, finfo in set_cfg.items():
            if key in ("description", "name") or not isinstance(finfo, dict):
                continue
            if not finfo.get("enabled", True):
                continue

            fid = finfo.get("family", key)
            base_id = finfo.get("base", "")
            instruct_id = finfo.get("instruct", "")
            adapter_name = finfo.get("adapter", fid)
            scale = finfo.get("scale", "")
            role = finfo.get("role", "replication")
            family_name = finfo.get("family_name", fid.capitalize())

            base_rev = finfo.get("base_revision", None)
            inst_rev = finfo.get("instruct_revision", None)

            fam_config = ModelFamilyConfig(
                family_id=fid,
                family_name=family_name,
                scale=scale,
                base_model=ModelSpec(model_id=base_id, format="plain", revision=base_rev),
                instruct_model=ModelSpec(model_id=instruct_id, format="chat", revision=inst_rev),
                adapter=adapter_name,
                role=role,
                enabled=True,
            )
            families[fid] = fam_config
            # 大文字小文字両方でアクセスできるように登録
            families[fid.capitalize()] = fam_config

    # 2. 旧構造 families からのロード（後方互換）
    elif "families" in data:
        for fid, finfo in data["families"].items():
            models = finfo.get("models", {})
            base_id = models.get("base", {}).get("model_id", "")
            inst_id = models.get("instruct", {}).get("model_id", "")
            fam_config = ModelFamilyConfig(
                family_id=fid,
                family_name=finfo.get("family_name", fid),
                scale=finfo.get("scale", ""),
                base_model=ModelSpec(model_id=base_id, format="plain"),
                instruct_model=ModelSpec(model_id=inst_id, format="chat"),
                adapter=finfo.get("architecture", fid.lower()),
                role=finfo.get("role", "replication"),
                num_layers=finfo.get("num_layers"),
                hidden_dim=finfo.get("hidden_dim"),
            )
            families[fid] = fam_config
            families[fid.lower()] = fam_config

    return families


def _slug_model_prefix(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace(".", "_").replace("-", "_")


def resolve_single_model_from_args(
    args,
    config_path: Optional[Path] = None,
) -> Tuple[str, str]:
    """
    単一モデル実行用 (V1 Primary 等) に (model_id, model_prefix) を解決する。

    優先順位:
      1. --family / --base-model / --instruct-model によるレジストリ解決
      2. 明示的な --model-id

    どちらも無い場合は例外を送出する。Qwen 等への静かな default は持たない。
    """
    family = getattr(args, "family", None)
    base_override = getattr(args, "base_model", None)
    inst_override = getattr(args, "instruct_model", None)
    model_id = getattr(args, "model_id", None)
    model_prefix = getattr(args, "model_prefix", None)
    is_instruct = bool(getattr(args, "is_instruct", False))

    if family or base_override or inst_override:
        target_models = resolve_models_from_args(args, config_path=config_path)
        if not target_models:
            raise KeyError("No models resolved from configs/models.yaml for the given selection.")
        if len(target_models) != 1:
            raise ValueError(
                "Multiple families resolved. Specify --family or --model-id to select exactly one model."
            )
        cfg = next(iter(target_models.values()))
        resolved_id = cfg.instruct_model.model_id if is_instruct else cfg.base_model.model_id
        resolved_prefix = model_prefix or f"{cfg.family_id}_{'instruct' if is_instruct else 'base'}"
        return resolved_id, resolved_prefix

    if model_id:
        resolved_prefix = model_prefix or _slug_model_prefix(model_id)
        return model_id, resolved_prefix

    raise ValueError(
        "Standalone execution requires --model-id or --family. "
        "Silent default to Qwen/Qwen2.5-1.5B-Instruct is disabled; "
        "use configs/models.yaml as the sole model source of truth."
    )


def resolve_instruct_target_from_args(
    args,
    config_path: Optional[Path] = None,
    fallback_family: Optional[str] = None,
) -> Tuple[str, str]:
    """
    (family_id, instruct_model_id) をレジストリから解決する。

    family が見つからない場合はハードコード Qwen ID へ落とさず KeyError を送出する。
    """
    target_models = resolve_models_from_args(args, config_path)
    family_filter = getattr(args, "family", None) or fallback_family
    if family_filter:
        fam_key = family_filter.lower()
        cfg = target_models.get(fam_key)
        if cfg is None:
            raise KeyError(
                f"Family '{family_filter}' not found in model-set "
                f"'{getattr(args, 'model_set', 'primary_small')}'. "
                f"Available: {list(target_models.keys())}"
            )
        return cfg.family_id, cfg.instruct_model.model_id
    if not target_models:
        raise KeyError("No models resolved from registry. Check --model-set and configs/models.yaml.")
    cfg = next(iter(target_models.values()))
    return cfg.family_id, cfg.instruct_model.model_id


class ModelRegistry:
    def __init__(self, config_path: Optional[Path] = None, model_set: str = "primary_small"):
        if config_path is None:
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parents[2]
            config_path = project_root / "configs" / "models.yaml"

        self.config_path = Path(config_path)
        self.model_set = model_set
        self.families: Dict[str, ModelFamilyConfig] = {}
        self._load()

    def _load(self):
        self.families = load_model_set(self.config_path, model_set=self.model_set)

    def get_family_by_model_id(self, model_id: str) -> ModelFamilyConfig:
        """
        HuggingFace model ID (例: 'Qwen/Qwen2.5-1.5B-Instruct') から ModelFamilyConfig を逆引きする。
        """
        # 1. 完全一致
        for cfg in self.families.values():
            if (cfg.instruct_model and cfg.instruct_model.model_id == model_id) or \
               (cfg.base_model and cfg.base_model.model_id == model_id):
                return cfg

        # 2. 小文字・部分一致
        clean_target = model_id.lower().strip()
        for cfg in self.families.values():
            inst_id = (cfg.instruct_model.model_id or "").lower()
            base_id = (cfg.base_model.model_id or "").lower()
            if clean_target == inst_id or clean_target == base_id:
                return cfg
            if clean_target in inst_id or clean_target in base_id:
                return cfg

        raise KeyError(
            f"No family found matching model_id '{model_id}' in set '{self.model_set}'. "
            f"Available families: {self.list_families()}"
        )

    def get_family(self, family_id: str) -> ModelFamilyConfig:
        if family_id in self.families:
            return self.families[family_id]
        if family_id.lower() in self.families:
            return self.families[family_id.lower()]
        if family_id.capitalize() in self.families:
            return self.families[family_id.capitalize()]

        # model_id による逆引きフォールバック
        try:
            return self.get_family_by_model_id(family_id)
        except KeyError:
            pass

        raise KeyError(f"Family or model ID '{family_id}' not found in set '{self.model_set}'. Available: {self.list_families()}")

    def list_families(self) -> List[str]:
        # 重複（小文字／大文字エイリアス）を除外してユニークな family_id を返す
        unique_ids = []
        for fid, cfg in self.families.items():
            if cfg.family_id not in unique_ids:
                unique_ids.append(cfg.family_id)
        return unique_ids

    def get_relative_depth(self, family_id: str, layer: int) -> float:
        num_layers = self.get_family(family_id).num_layers
        if num_layers <= 1:
            return 0.0
        return layer / (num_layers - 1)


_GLOBAL_REGISTRY: Optional[ModelRegistry] = None


def get_registry(config_path: Optional[Path] = None, model_set: str = "primary_small") -> ModelRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None or config_path is not None or _GLOBAL_REGISTRY.model_set != model_set:
        _GLOBAL_REGISTRY = ModelRegistry(config_path, model_set=model_set)
    return _GLOBAL_REGISTRY


def add_model_selection_args(parser):
    """CLI ArgumentParser にモデル選択・オーバーライド引数を追加する共通ヘルパー"""
    group = parser.add_argument_group("Model Selection & Registry")
    group.add_argument(
        "--model-set",
        type=str,
        default="primary_small",
        choices=["primary_small", "scale_validation"],
        help="Model cohort to evaluate (default: primary_small)",
    )
    group.add_argument(
        "--family",
        type=str,
        default=None,
        help="Optional specific family filter (e.g., qwen, llama, gemma, olmo, mistral)",
    )
    group.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override base model ID via CLI",
    )
    group.add_argument(
        "--instruct-model",
        type=str,
        default=None,
        help="Override instruct model ID via CLI",
    )
    return parser


def resolve_models_from_args(
    args,
    config_path: Optional[Path] = None,
) -> Dict[str, ModelFamilyConfig]:
    """
    CLI引数（args）に基づいてモデル設定を解決する。
    優先順位: CLI (--base-model / --instruct-model) > YAML 設定 > デフォルト
    """
    model_set = getattr(args, "model_set", "primary_small")
    family_filter = getattr(args, "family", None)
    base_override = getattr(args, "base_model", None)
    inst_override = getattr(args, "instruct_model", None)

    families = load_model_set(config_path, model_set=model_set)

    # 特定 family でのフィルタリング
    if family_filter:
        fam_key = family_filter.lower()
        if fam_key not in families and fam_key.capitalize() not in families:
            raise KeyError(f"Family '{family_filter}' not found in set '{model_set}'. Available: {list(families.keys())}")
        target_cfg = families.get(fam_key) or families.get(fam_key.capitalize())
        families = {target_cfg.family_id: target_cfg}
    else:
        # 重複エイリアスを排除して family_id をキーとする辞書に正規化
        unique_fams = {}
        for cfg in families.values():
            unique_fams[cfg.family_id] = cfg
        families = unique_fams

    # CLI によるモデルID override
    if base_override or inst_override:
        if not family_filter:
            raise ValueError(
                "CLI model override (--base-model or --instruct-model) requires explicit --family argument "
                "to prevent ambiguous or unintended overwriting across multiple model families."
            )
        for fid, cfg in families.items():
            if base_override:
                cfg.base_model = ModelSpec(model_id=base_override, format=cfg.base_model.format)
            if inst_override:
                cfg.instruct_model = ModelSpec(model_id=inst_override, format=cfg.instruct_model.format)

    return families
