import argparse
import pytest
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from affective_empathy_eval.models.registry import (
    load_model_set,
    get_registry,
    resolve_architecture_dims,
    add_model_selection_args,
    resolve_models_from_args,
    resolve_single_model_from_args,
    resolve_instruct_target_from_args,
)
from affective_empathy_eval.run import PRODUCTION_STAGE_ORDER
from affective_empathy_eval.data import describe_loaded_frame
from affective_empathy_eval.models.adapters import (
    get_model_adapter,
    LlamaFamilyAdapter,
    GemmaAdapter,
    Olmo2Adapter,
)


def test_resolve_architecture_dims():
    layers, dim = resolve_architecture_dims("Qwen/Qwen2.5-1.5B")
    assert layers == 28
    assert dim == 1536

    layers, dim = resolve_architecture_dims("meta-llama/Llama-3.2-1B")
    assert layers == 16
    assert dim == 2048

    layers, dim = resolve_architecture_dims("google/gemma-3-1b-it")
    assert layers == 26
    assert dim == 1152

    layers, dim = resolve_architecture_dims("allenai/OLMo-2-0425-1B")
    assert layers == 16
    assert dim == 2048

    layers, dim = resolve_architecture_dims("mistralai/Mistral-7B-Instruct-v0.3")
    assert layers == 32
    assert dim == 4096


def test_load_primary_small():
    models = load_model_set(model_set="primary_small")
    assert "qwen" in models
    assert "llama" in models
    assert "gemma" in models
    assert "olmo" in models

    qwen = models["qwen"]
    assert qwen.num_layers == 28
    assert qwen.hidden_dim == 1536
    assert qwen.base_model.model_id == "Qwen/Qwen2.5-1.5B"
    assert qwen.instruct_model.model_id == "Qwen/Qwen2.5-1.5B-Instruct"

    llama = models["llama"]
    assert llama.num_layers == 16
    assert llama.hidden_dim == 2048

    olmo = models["olmo"]
    assert olmo.num_layers == 16
    assert olmo.hidden_dim == 2048


def test_load_scale_validation():
    models = load_model_set(model_set="scale_validation")
    assert "mistral" in models
    mistral = models["mistral"]
    assert mistral.num_layers == 32
    assert mistral.hidden_dim == 4096
    assert "v0.3" in mistral.base_model.model_id
    assert "v0.3" in mistral.instruct_model.model_id


def test_resolve_models_from_args():
    parser = argparse.ArgumentParser()
    add_model_selection_args(parser)

    # 1. デフォルト
    args = parser.parse_args([])
    models = resolve_models_from_args(args)
    assert len(models) == 4
    assert "qwen" in models

    # 2. ファミリー絞り込み
    args = parser.parse_args(["--family", "qwen"])
    models = resolve_models_from_args(args)
    assert len(models) == 1
    assert "qwen" in models

    # 3. CLI override (family 指定あり -> 成功)
    args = parser.parse_args([
        "--family", "gemma",
        "--base-model", "custom/gemma-base",
        "--instruct-model", "custom/gemma-it",
    ])
    models = resolve_models_from_args(args)
    assert models["gemma"].base_model.model_id == "custom/gemma-base"
    assert models["gemma"].instruct_model.model_id == "custom/gemma-it"

    # 4. CLI override (family 指定なし -> ValueError)
    args_invalid = parser.parse_args(["--base-model", "custom/base"])
    with pytest.raises(ValueError, match="requires explicit --family"):
        resolve_models_from_args(args_invalid)


def test_production_stage_order_is_behavioral_then_v1_v2_v3():
    assert PRODUCTION_STAGE_ORDER == ("behavioral", "v1", "v2", "v3")


def test_resolve_single_model_requires_model_id_or_family():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--model-prefix", type=str, default=None)
    parser.add_argument("--is-instruct", action="store_true")
    add_model_selection_args(parser)

    with pytest.raises(ValueError, match="requires --model-id or --family"):
        resolve_single_model_from_args(parser.parse_args([]))

    args = parser.parse_args(["--family", "llama", "--is-instruct"])
    model_id, prefix = resolve_single_model_from_args(args)
    assert model_id == "meta-llama/Llama-3.2-1B-Instruct"
    assert prefix == "llama_instruct"


def test_resolve_instruct_target_unknown_family_raises_keyerror():
    parser = argparse.ArgumentParser()
    add_model_selection_args(parser)
    args = parser.parse_args(["--family", "not_a_real_family"])
    with pytest.raises(KeyError, match="not_a_real_family"):
        resolve_instruct_target_from_args(args)


def test_describe_loaded_frame_reports_actual_counts():
    df = pd.DataFrame({
        "pair_id": [1, 1, 2, 2],
        "stimulus_id": ["a", "b", "c", "d"],
        "split": ["clinical", "neutral", "clinical", "neutral"],
    })
    text = describe_loaded_frame(df, "AIPsy", "dummy.csv")
    assert "n_rows=4" in text
    assert "n_unique_pair_id=2" in text
    assert "n_unique_stimulus_id=4" in text


def test_unknown_model_dimension_raises_error():
    with pytest.raises(ValueError, match="Unable to resolve architecture dimensions"):
        resolve_architecture_dims("completely_unknown_nonexistent_model_xyz_123")


# --- Mock Models for Adapter Testing ---

class MockLlamaLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(16, 16)
        self.mlp = nn.Linear(16, 16)


class MockLlamaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([MockLlamaLayer() for _ in range(4)])
        self.model.norm = nn.LayerNorm(16)
        self.lm_head = nn.Linear(16, 32)


class MockGemmaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([MockLlamaLayer() for _ in range(4)])
        self.model.norm = nn.LayerNorm(16)
        self.lm_head = nn.Linear(16, 32)


class MockOlmoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.blocks = nn.ModuleList([MockLlamaLayer() for _ in range(4)])
        self.transformer.ln_f = nn.LayerNorm(16)
        self.transformer.ff_out = nn.Linear(16, 32)


def test_adapters():
    llama_model = MockLlamaModel()
    llama_adapter = get_model_adapter(llama_model, adapter_name="llama")
    assert isinstance(llama_adapter, LlamaFamilyAdapter)
    assert llama_adapter.get_num_layers() == 4
    assert llama_adapter.get_hook_target(1, "attn") is llama_model.model.layers[1].self_attn

    gemma_model = MockGemmaModel()
    gemma_adapter = get_model_adapter(gemma_model, adapter_name="gemma3")
    assert isinstance(gemma_adapter, GemmaAdapter)
    assert gemma_adapter.get_num_layers() == 4

    olmo_model = MockOlmoModel()
    olmo_adapter = get_model_adapter(olmo_model, adapter_name="olmo2")
    assert isinstance(olmo_adapter, Olmo2Adapter)
    assert olmo_adapter.get_num_layers() == 4
    assert olmo_adapter.get_final_norm() is olmo_model.transformer.ln_f


def test_registry_get_family_by_model_id():
    """V3本番経路で渡されるHF model IDから正確にfamily configを逆引きできることを検証"""
    reg = get_registry(model_set="primary_small")
    
    # 4 family x 2 (base & instruct) = 8モデルすべてでテスト
    test_cases = [
        ("Qwen/Qwen2.5-1.5B", "qwen"),
        ("Qwen/Qwen2.5-1.5B-Instruct", "qwen"),
        ("meta-llama/Llama-3.2-1B", "llama"),
        ("meta-llama/Llama-3.2-1B-Instruct", "llama"),
        ("google/gemma-3-1b-pt", "gemma"),
        ("google/gemma-3-1b-it", "gemma"),
        ("allenai/OLMo-2-0425-1B", "olmo"),
        ("allenai/OLMo-2-0425-1B-Instruct", "olmo"),
    ]
    for model_id, expected_fam in test_cases:
        cfg1 = reg.get_family_by_model_id(model_id)
        assert cfg1.family_id.lower() == expected_fam, f"Failed get_family_by_model_id for {model_id}"
        cfg2 = reg.get_family(model_id)
        assert cfg2.family_id.lower() == expected_fam, f"Failed fallback get_family for {model_id}"

    # scale_validation の Mistral 7B も検証
    reg_scale = get_registry(model_set="scale_validation")
    cfg_mistral_base = reg_scale.get_family_by_model_id("mistralai/Mistral-7B-v0.3")
    assert cfg_mistral_base.family_id.lower() == "mistral"
    cfg_mistral_inst = reg_scale.get_family("mistralai/Mistral-7B-Instruct-v0.3")
    assert cfg_mistral_inst.family_id.lower() == "mistral"


def test_v3_entrypoints_with_mock_model():
    """V3スクリプトが実モデルIDを受け取った時の初期化・アダプタ取得経路を検証"""
    reg = get_registry(model_set="primary_small")
    qwen_id = "Qwen/Qwen2.5-1.5B-Instruct"
    fam_cfg = reg.get_family_by_model_id(qwen_id)
    
    # Mock model でアダプタ解決
    mock_model = MockLlamaModel()
    adapter = get_model_adapter(mock_model, fam_cfg)
    assert adapter is not None
    assert adapter.get_num_layers() == 4
