from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from .adapters import ModelAdapter



class HookPoint(Enum):
    PRE_ATTN_RESID = "pre_attn_resid"      # レイヤー入力（ブロック前残差）
    POST_ATTN_RESID = "post_attn_resid"    # アテンション後残差
    POST_MLP_RESID = "post_mlp_resid"      # MLP後残差（ブロック出力）
    ATTN_OUT = "attn_out"                  # アテンション単体出力（残差加算前）
    MLP_OUT = "mlp_out"                    # MLP単体出力（残差加算前）


class ActivationHookManager:
    """
    PyTorch Forward Hook を管理し、活性化の記録・パッチング・射影除去・方向注入を制御するマネージャー
    コンテキストマネージャーとして使用可能
    """

    def __init__(self, adapter: ModelAdapter):
        self.adapter = adapter
        self.handles: List[Any] = []
        self.captured_activations: Dict[str, torch.Tensor] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_all()

    def remove_all(self):
        """登録されているすべてのフックを解除"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def register_capture_hook(
        self,
        layer_idx: int,
        hook_point: HookPoint = HookPoint.POST_MLP_RESID,
        token_indices: Optional[Union[int, List[int], slice]] = None,
        key: Optional[str] = None,
    ):
        """
        特定層・特定ポイントの活性化を保存するフックを登録
        """
        hook_key = key or f"layer_{layer_idx}_{hook_point.value}"
        target_module, is_input = self._resolve_target(layer_idx, hook_point)

        def hook_fn(module, args, output):
            tensor = args[0] if is_input else output
            if isinstance(tensor, tuple):
                tensor = tensor[0]

            # token_indices のスライシング
            if token_indices is not None:
                if isinstance(token_indices, int):
                    tensor = tensor[:, token_indices : token_indices + 1, :]
                elif isinstance(token_indices, list):
                    tensor = tensor[:, token_indices, :]
                elif isinstance(token_indices, slice):
                    tensor = tensor[:, token_indices, :]

            self.captured_activations[hook_key] = tensor.detach().clone()

        handle = target_module.register_forward_hook(hook_fn)
        self.handles.append(handle)

    def register_patch_hook(
        self,
        layer_idx: int,
        patch_tensor: torch.Tensor,
        token_indices: Optional[Union[int, List[int], slice]] = None,
        hook_point: HookPoint = HookPoint.POST_MLP_RESID,
    ):
        """
        特定層・特定トークン位置の活性化を patch_tensor で置換するフックを登録
        """
        target_module, is_input = self._resolve_target(layer_idx, hook_point)

        def hook_fn(module, args, output):
            is_tuple = isinstance(output, tuple)
            out_tensor = output[0] if is_tuple else output

            new_tensor = out_tensor.clone()
            target_dev = out_tensor.device
            target_dtype = out_tensor.dtype
            patch = patch_tensor.to(device=target_dev, dtype=target_dtype)

            if token_indices is None:
                new_tensor = patch
            elif isinstance(token_indices, int):
                new_tensor[:, token_indices : token_indices + 1, :] = patch
            elif isinstance(token_indices, list):
                new_tensor[:, token_indices, :] = patch
            elif isinstance(token_indices, slice):
                new_tensor[:, token_indices, :] = patch

            if is_tuple:
                return (new_tensor,) + output[1:]
            return new_tensor

        handle = target_module.register_forward_hook(hook_fn)
        self.handles.append(handle)

    def register_direction_intervention_hook(
        self,
        layer_idx: int,
        direction: Union[torch.Tensor, np.ndarray],
        alpha: float,
        hidden_std: float = 1.0,
        token_indices: Optional[Union[int, List[int], slice]] = None,
        hook_point: HookPoint = HookPoint.POST_MLP_RESID,
        mode: str = "inject",
    ):
        """
        V3 統一介入オペレータ:
          mode="inject":  h' = h + alpha * hidden_std * unit(direction)
          mode="replace": h' = alpha * hidden_std * unit(direction)

        ここで:
          - unit(direction) は単位ノルムの方向ベクトル d_hat
          - hidden_std は対象 layer/site での hidden activation scale (sigma_h)
          - alpha は無次元介入強度
        """
        assert mode in ("inject", "replace"), f"mode must be 'inject' or 'replace', got '{mode}'"
        target_module, is_input = self._resolve_target(layer_idx, hook_point)

        if isinstance(direction, np.ndarray):
            dir_tensor = torch.from_numpy(direction).float()
        else:
            dir_tensor = direction.clone().detach().float()

        def hook_fn(module, args, output):
            is_tuple = isinstance(output, tuple)
            out_tensor = output[0] if is_tuple else output

            new_tensor = out_tensor.clone()
            d = dir_tensor.to(device=out_tensor.device, dtype=out_tensor.dtype)
            d_norm = torch.norm(d, p=2)
            d_hat = d / (d_norm + 1e-12) if d_norm > 0 else torch.zeros_like(d)

            # scale は alpha 側で乗算 (sigma_h * alpha)
            delta = float(alpha) * float(hidden_std) * d_hat
            if delta.ndim == 1:
                delta = delta.view(1, 1, -1)
            elif delta.ndim == 2:
                delta = delta.unsqueeze(1)

            if token_indices is None:
                if mode == "inject":
                    new_tensor = new_tensor + delta
                else:
                    new_tensor = torch.broadcast_to(delta, new_tensor.shape).clone()
            elif isinstance(token_indices, int):
                if mode == "inject":
                    new_tensor[:, token_indices : token_indices + 1, :] += delta
                else:
                    new_tensor[:, token_indices : token_indices + 1, :] = delta
            elif isinstance(token_indices, list):
                if mode == "inject":
                    new_tensor[:, token_indices, :] += delta
                else:
                    new_tensor[:, token_indices, :] = delta
            elif isinstance(token_indices, slice):
                if mode == "inject":
                    new_tensor[:, token_indices, :] += delta
                else:
                    new_tensor[:, token_indices, :] = delta

            if is_tuple:
                return (new_tensor,) + output[1:]
            return new_tensor

        handle = target_module.register_forward_hook(hook_fn)
        self.handles.append(handle)
        return handle

    def register_direction_injection_hook(
        self,
        layer_idx: int,
        direction: Union[torch.Tensor, np.ndarray],
        alpha: float,
        token_indices: Optional[Union[int, List[int], slice]] = None,
        hook_point: HookPoint = HookPoint.POST_MLP_RESID,
        hidden_std: float = 1.0,
        mode: str = "inject",
    ):
        """
        後方互換・短縮メソッド: register_direction_intervention_hook を呼び出す。
        """
        return self.register_direction_intervention_hook(
            layer_idx=layer_idx,
            direction=direction,
            alpha=alpha,
            hidden_std=hidden_std,
            token_indices=token_indices,
            hook_point=hook_point,
            mode=mode,
        )


    def register_centered_projection_removal_hook(
        self,
        layer_idx: int,
        direction: torch.Tensor,
        mean_vector: Optional[torch.Tensor] = None,
        token_indices: Optional[Union[int, List[int], slice]] = None,
        hook_point: HookPoint = HookPoint.POST_MLP_RESID,
    ):
        """
        Centered Projection Removal:
        h' = h - [(h - mu)^T d_hat] d_hat
        """
        target_module, is_input = self._resolve_target(layer_idx, hook_point)

        def hook_fn(module, args, output):
            is_tuple = isinstance(output, tuple)
            out_tensor = output[0] if is_tuple else output

            new_tensor = out_tensor.clone()
            dev = out_tensor.device
            dtype = out_tensor.dtype

            d = direction.to(device=dev, dtype=dtype)
            d_hat = d / torch.norm(d, p=2)
            if d_hat.ndim == 1:
                d_hat = d_hat.view(-1, 1)  # (H, 1)

            mu = mean_vector.to(device=dev, dtype=dtype) if mean_vector is not None else torch.zeros_like(d_hat).squeeze()
            if mu.ndim == 1:
                mu = mu.view(1, 1, -1)

            # 対象トークン領域の抽出
            if token_indices is None:
                sub_h = new_tensor
            elif isinstance(token_indices, int):
                sub_h = new_tensor[:, token_indices : token_indices + 1, :]
            elif isinstance(token_indices, list):
                sub_h = new_tensor[:, token_indices, :]
            elif isinstance(token_indices, slice):
                sub_h = new_tensor[:, token_indices, :]

            # 中心化偏差: h_c = sub_h - mu
            h_c = sub_h - mu  # (B, T, H)
            # 射影成分: proj = (h_c @ d_hat) @ d_hat^T
            # h_c @ d_hat: (B, T, 1)
            scalar_proj = torch.matmul(h_c, d_hat)  # (B, T, 1)
            proj_vector = scalar_proj * d_hat.squeeze().view(1, 1, -1)  # (B, T, H)

            removed_h = sub_h - proj_vector

            if token_indices is None:
                new_tensor = removed_h
            elif isinstance(token_indices, int):
                new_tensor[:, token_indices : token_indices + 1, :] = removed_h
            elif isinstance(token_indices, list):
                new_tensor[:, token_indices, :] = removed_h
            elif isinstance(token_indices, slice):
                new_tensor[:, token_indices, :] = removed_h

            if is_tuple:
                return (new_tensor,) + output[1:]
            return new_tensor

        handle = target_module.register_forward_hook(hook_fn)
        self.handles.append(handle)

    def register_subspace_removal_hook(
        self,
        layer_idx: int,
        orth_basis_q: torch.Tensor,  # (H, K)
        mean_vector: Optional[torch.Tensor] = None,
        token_indices: Optional[Union[int, List[int], slice]] = None,
        hook_point: HookPoint = HookPoint.POST_MLP_RESID,
    ):
        """
        2D Affective Subspace Removal:
        h' = h - Q Q^T (h - mu)
        """
        target_module, is_input = self._resolve_target(layer_idx, hook_point)

        def hook_fn(module, args, output):
            is_tuple = isinstance(output, tuple)
            out_tensor = output[0] if is_tuple else output

            new_tensor = out_tensor.clone()
            dev = out_tensor.device
            dtype = out_tensor.dtype

            Q = orth_basis_q.to(device=dev, dtype=dtype)  # (H, K)
            mu = mean_vector.to(device=dev, dtype=dtype) if mean_vector is not None else torch.zeros(Q.shape[0], device=dev, dtype=dtype)
            if mu.ndim == 1:
                mu = mu.view(1, 1, -1)

            if token_indices is None:
                sub_h = new_tensor
            elif isinstance(token_indices, int):
                sub_h = new_tensor[:, token_indices : token_indices + 1, :]
            elif isinstance(token_indices, list):
                sub_h = new_tensor[:, token_indices, :]
            elif isinstance(token_indices, slice):
                sub_h = new_tensor[:, token_indices, :]

            h_c = sub_h - mu  # (B, T, H)
            # h_c @ Q: (B, T, K)
            # (h_c @ Q) @ Q^T: (B, T, H)
            proj_vector = torch.matmul(torch.matmul(h_c, Q), Q.t())  # (B, T, H)
            removed_h = sub_h - proj_vector

            if token_indices is None:
                new_tensor = removed_h
            elif isinstance(token_indices, int):
                new_tensor[:, token_indices : token_indices + 1, :] = removed_h
            elif isinstance(token_indices, list):
                new_tensor[:, token_indices, :] = removed_h
            elif isinstance(token_indices, slice):
                new_tensor[:, token_indices, :] = removed_h

            if is_tuple:
                return (new_tensor,) + output[1:]
            return new_tensor

        handle = target_module.register_forward_hook(hook_fn)
        self.handles.append(handle)

    def _resolve_target(self, layer_idx: int, hook_point: HookPoint) -> Tuple[nn.Module, bool]:
        if hook_point == HookPoint.PRE_ATTN_RESID:
            return self.adapter.get_layers()[layer_idx], True
        elif hook_point == HookPoint.POST_MLP_RESID:
            return self.adapter.get_layers()[layer_idx], False
        elif hook_point == HookPoint.ATTN_OUT:
            return self.adapter.get_attn_module(layer_idx), False
        elif hook_point == HookPoint.MLP_OUT:
            return self.adapter.get_mlp_module(layer_idx), False
        elif hook_point == HookPoint.POST_ATTN_RESID:
            # post_attn_resid は多くのモデルで残差ストリーム加算後なので、
            # MLPモジュールの input (PRE_MLP) と等価
            return self.adapter.get_mlp_module(layer_idx), True
        else:
            raise ValueError(f"Unsupported hook point: {hook_point}")


def apply_direction_intervention(
    adapter: ModelAdapter,
    layer_idx: int,
    direction: Union[torch.Tensor, np.ndarray],
    alpha: float,
    hidden_std: float = 1.0,
    token_position: Optional[Union[int, List[int], slice]] = None,
    mode: str = "inject",
    hook_point: HookPoint = HookPoint.POST_MLP_RESID,
) -> ActivationHookManager:
    """
    共通介入コンテキストマネージャーを作成・登録して返す。
    使用例:
        with apply_direction_intervention(adapter, layer_idx, direction, alpha, hidden_std, token_pos, mode="inject"):
            outputs = model(...)
    """
    hook_mgr = ActivationHookManager(adapter)
    hook_mgr.register_direction_intervention_hook(
        layer_idx=layer_idx,
        direction=direction,
        alpha=alpha,
        hidden_std=hidden_std,
        token_indices=token_position,
        hook_point=hook_point,
        mode=mode,
    )
    return hook_mgr

