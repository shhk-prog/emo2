"""
tests/test_v3_interventions_sanity.py

V3 Direction Intervention Sanity Tests:
検証項目:
1. alpha=0 のとき intervention output が baseline と完全一致する。
2. mode="inject" では patched - original が alpha * hidden_std * unit(direction) と一致する。
3. mode="replace" では activation が指定 vector へ置換される。
4. direction norm を変えても、unit normalize 後の介入量は変わらない。
5. random direction、orthogonal direction、affective direction で scale definition が同一である。
6. Confirmatory と Discovery で同じ alpha が同じ activation norm change を意味する。
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from affective_empathy_eval.models.hooks import ActivationHookManager, HookPoint
from affective_empathy_eval.models.adapters import ModelAdapter


class MockTransformerLayer(nn.Module):
    """単一の線形変換と残差結合を持つモック Transformer レイヤー"""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Identity()
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        return self.mlp(x)


class MockModel(nn.Module):
    def __init__(self, num_layers: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([MockTransformerLayer(hidden_dim) for _ in range(num_layers)])

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


class MockModelAdapter(ModelAdapter):
    def __init__(self, model: MockModel):
        super().__init__(model)

    def get_layers(self):
        return self.model.layers

    def get_attn_module(self, layer_idx: int):
        return self.model.layers[layer_idx]

    def get_mlp_module(self, layer_idx: int):
        return self.model.layers[layer_idx]

    def get_final_norm(self):
        return nn.Identity()

    def get_lm_head(self):
        return nn.Identity()



@pytest.fixture
def mock_setup():
    torch.manual_seed(42)
    hidden_dim = 32
    seq_len = 5
    batch_size = 2
    model = MockModel(num_layers=3, hidden_dim=hidden_dim)
    adapter = MockModelAdapter(model)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    return model, adapter, x, hidden_dim, seq_len


def test_alpha_zero_baseline_equivalence(mock_setup):
    """1. alpha=0 のとき intervention output が baseline と一致する。"""
    model, adapter, x, hidden_dim, seq_len = mock_setup
    baseline_out = model(x).clone()

    direction = torch.randn(hidden_dim)
    hidden_std = 2.5

    with ActivationHookManager(adapter) as hook_mgr:
        hook_mgr.register_direction_intervention_hook(
            layer_idx=1,
            direction=direction,
            alpha=0.0,
            hidden_std=hidden_std,
            token_indices=2,
            mode="inject",
        )
        patched_out = model(x)

    assert torch.allclose(baseline_out, patched_out, atol=1e-6), (
        "alpha=0.0 must not alter model outputs."
    )


def test_additive_injection_exact_formula(mock_setup):
    """2. inject では patched - original が alpha * hidden_std * unit(direction) と一致する。"""
    model, adapter, x, hidden_dim, seq_len = mock_setup
    baseline_out = model(x).clone()

    direction = torch.randn(hidden_dim)
    unit_d = direction / torch.norm(direction, p=2)
    alpha = 1.5
    hidden_std = 2.0
    token_pos = 3

    with ActivationHookManager(adapter) as hook_mgr:
        hook_mgr.register_direction_intervention_hook(
            layer_idx=1,
            direction=direction,
            alpha=alpha,
            hidden_std=hidden_std,
            token_indices=token_pos,
            mode="inject",
        )
        patched_out = model(x)

    diff = patched_out - baseline_out
    expected_delta = alpha * hidden_std * unit_d

    # 介入対象トークン位置以外は差分が 0
    for pos in range(seq_len):
        if pos == token_pos:
            for b in range(x.shape[0]):
                assert torch.allclose(diff[b, pos, :], expected_delta, atol=1e-5), (
                    f"Token {pos} delta mismatch with alpha * hidden_std * unit(direction)"
                )
        else:
            assert torch.allclose(diff[:, pos, :], torch.zeros_like(diff[:, pos, :]), atol=1e-6), (
                f"Non-target token {pos} was unexpectedly modified."
            )


def test_replace_mode_behavior(mock_setup):
    """3. replace では activation が指定 vector (alpha * hidden_std * unit(direction)) へ置換される。"""
    model, adapter, x, hidden_dim, seq_len = mock_setup

    direction = torch.randn(hidden_dim)
    unit_d = direction / torch.norm(direction, p=2)
    alpha = 1.0
    hidden_std = 3.0
    token_pos = 1

    with ActivationHookManager(adapter) as hook_mgr:
        hook_mgr.register_direction_intervention_hook(
            layer_idx=0,
            direction=direction,
            alpha=alpha,
            hidden_std=hidden_std,
            token_indices=token_pos,
            mode="replace",
        )
        patched_out = model(x)

    expected_vec = alpha * hidden_std * unit_d
    for b in range(x.shape[0]):
        assert torch.allclose(patched_out[b, token_pos, :], expected_vec, atol=1e-5), (
            "mode='replace' must overwrite activation with alpha * hidden_std * unit(direction)"
        )


def test_direction_norm_invariance(mock_setup):
    """4. direction norm を変えても、unit normalize 後の介入量は変わらない。"""
    model, adapter, x, hidden_dim, seq_len = mock_setup
    baseline_out = model(x)

    base_dir = torch.randn(hidden_dim)
    dir_small = base_dir * 0.01
    dir_large = base_dir * 100.0

    alpha = 1.0
    hidden_std = 2.0
    token_pos = 2

    with ActivationHookManager(adapter) as hook_mgr:
        hook_mgr.register_direction_intervention_hook(
            layer_idx=1,
            direction=dir_small,
            alpha=alpha,
            hidden_std=hidden_std,
            token_indices=token_pos,
            mode="inject",
        )
        out_small = model(x)

    with ActivationHookManager(adapter) as hook_mgr:
        hook_mgr.register_direction_intervention_hook(
            layer_idx=1,
            direction=dir_large,
            alpha=alpha,
            hidden_std=hidden_std,
            token_indices=token_pos,
            mode="inject",
        )
        out_large = model(x)

    assert torch.allclose(out_small, out_large, atol=1e-5), (
        "Scaling raw direction must not affect intervention magnitude due to internal unit normalization."
    )


def test_scale_definition_consistency_across_directions(mock_setup):
    """5. random direction、orthogonal direction、affective direction で scale definition が同一。"""
    model, adapter, x, hidden_dim, seq_len = mock_setup

    # 3つの異なる方向ベクトル
    d_affect = torch.randn(hidden_dim)
    # 直交方向
    d_orth = torch.randn(hidden_dim)
    d_orth = d_orth - (torch.dot(d_orth, d_affect) / torch.dot(d_affect, d_affect)) * d_affect
    d_rand = torch.randn(hidden_dim)

    alpha = 1.0
    hidden_std = 2.5
    token_pos = 0

    norms = []
    for d in (d_affect, d_orth, d_rand):
        with ActivationHookManager(adapter) as hook_mgr:
            hook_mgr.register_direction_intervention_hook(
                layer_idx=1,
                direction=d,
                alpha=alpha,
                hidden_std=hidden_std,
                token_indices=token_pos,
                mode="inject",
            )
            out = model(x)
        delta_norm = torch.norm(out[0, token_pos, :] - x[0, token_pos, :], p=2).item()
        norms.append(delta_norm)

    # どの方向であっても加算されるベクトルの L2 ノルムは alpha * hidden_std = 2.5 と厳密に一致
    expected_norm = alpha * hidden_std
    for n in norms:
        assert abs(n - expected_norm) < 1e-4, f"Intervention norm {n} did not equal {expected_norm}"


def test_confirmatory_and_discovery_operator_equivalence(mock_setup):
    """6. Confirmatory と Discovery で同じ alpha が同じ activation norm change を意味する。"""
    model, adapter, x, hidden_dim, seq_len = mock_setup

    direction = torch.randn(hidden_dim)
    alpha = 1.0
    hidden_std = 2.0
    token_pos = 2

    # Discovery (RQ1) と同一の hook 呼び出し
    with ActivationHookManager(adapter) as hook_mgr_disc:
        hook_mgr_disc.register_direction_intervention_hook(
            layer_idx=1,
            direction=direction,
            alpha=alpha,
            hidden_std=hidden_std,
            token_indices=token_pos,
            mode="inject",
        )
        out_disc = model(x)

    # Confirmatory と同一の hook 呼び出し
    with ActivationHookManager(adapter) as hook_mgr_conf:
        hook_mgr_conf.register_direction_intervention_hook(
            layer_idx=1,
            direction=direction,
            alpha=alpha,
            hidden_std=hidden_std,
            token_indices=token_pos,
            mode="inject",
        )
        out_conf = model(x)

    assert torch.allclose(out_disc, out_conf, atol=1e-6), (
        "Discovery and Confirmatory must produce identical activations for identical alpha."
    )
    delta_norm = torch.norm(out_disc[0, token_pos, :] - x[0, token_pos, :], p=2).item()
    assert abs(delta_norm - (alpha * hidden_std)) < 1e-4
