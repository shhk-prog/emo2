import torch
from torch import nn

from affective_empathy_eval.models.adapters import LlamaFamilyAdapter
from affective_empathy_eval.models.hooks import ActivationHookManager, HookPoint


class DummyTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_attn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        h = x + self.self_attn(x)
        out = h + self.mlp(h)
        return out


class DummyModel(nn.Module):
    def __init__(self, hidden_dim: int = 16, num_layers: int = 4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyTransformerBlock(hidden_dim) for _ in range(num_layers)])
        self.model.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, 100, bias=False)

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        return self.lm_head(x)


def test_adapter_and_hooks():
    hidden_dim = 16
    num_layers = 4
    model = DummyModel(hidden_dim=hidden_dim, num_layers=num_layers)
    adapter = LlamaFamilyAdapter(model)

    assert adapter.get_num_layers() == num_layers
    assert adapter.get_final_norm() == model.model.norm

    # 1. Capture Hook
    with ActivationHookManager(adapter) as hook_mgr:
        hook_mgr.register_capture_hook(layer_idx=1, hook_point=HookPoint.POST_MLP_RESID)
        dummy_input = torch.randn(2, 5, hidden_dim)
        _ = model(dummy_input)

        captured = hook_mgr.captured_activations.get("layer_1_post_mlp_resid")
        assert captured is not None
        assert captured.shape == (2, 5, hidden_dim)

    # 2. Patch Hook
    with ActivationHookManager(adapter) as hook_mgr:
        patch_val = torch.ones(2, 1, hidden_dim) * 99.0
        hook_mgr.register_patch_hook(layer_idx=1, patch_tensor=patch_val, token_indices=2)
        hook_mgr.register_capture_hook(layer_idx=1, hook_point=HookPoint.POST_MLP_RESID)

        _ = model(dummy_input)
        captured = hook_mgr.captured_activations.get("layer_1_post_mlp_resid")
        assert torch.allclose(captured[:, 2:3, :], patch_val)

    # 3. Centered Projection Removal Hook
    with ActivationHookManager(adapter) as hook_mgr:
        direction = torch.zeros(hidden_dim)
        direction[0] = 1.0  # 最初の次元方向のみ
        mean_vec = torch.zeros(hidden_dim)

        hook_mgr.register_centered_projection_removal_hook(
            layer_idx=1,
            direction=direction,
            mean_vector=mean_vec,
        )
        hook_mgr.register_capture_hook(layer_idx=1, hook_point=HookPoint.POST_MLP_RESID)

        _ = model(dummy_input)
        captured = hook_mgr.captured_activations.get("layer_1_post_mlp_resid")
        # direction (0次元目) の成分が 0 になっていること
        assert torch.allclose(captured[..., 0], torch.zeros_like(captured[..., 0]), atol=1e-5)
