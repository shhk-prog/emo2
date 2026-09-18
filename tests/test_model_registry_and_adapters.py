import argparse
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from affective_empathy_eval.models.registry import (
    load_model_set,
    get_registry,
    resolve_architecture_dims,
    add_model_selection_args,
    resolve_models_from_args,
)
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

    # 3. CLI override
    args = parser.parse_args([
        "--family", "gemma",
        "--base-model", "custom/gemma-base",
        "--instruct-model", "custom/gemma-it",
    ])
    models = resolve_models_from_args(args)
    assert models["gemma"].base_model.model_id == "custom/gemma-base"
    assert models["gemma"].instruct_model.model_id == "custom/gemma-it"


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
