from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class ModelAdapter(ABC):
    """
    アーキテクチャ間の差異（Qwen, Llama, Gemma, Mistral）を吸収する共通アダプター
    """

    def __init__(self, model: nn.Module):
        self.model = model

    @abstractmethod
    def get_layers(self) -> nn.ModuleList:
        """トランスフォーマー層リストを返す"""
        pass

    def get_num_layers(self) -> int:
        return len(self.get_layers())

    @abstractmethod
    def get_attn_module(self, layer_idx: int) -> nn.Module:
        """特定層のアテンションモジュールを返す"""
        pass

    @abstractmethod
    def get_mlp_module(self, layer_idx: int) -> nn.Module:
        """特定層のMLPモジュールを返す"""
        pass

    @abstractmethod
    def get_final_norm(self) -> nn.Module:
        """最終正規化層（RMSNorm / LayerNorm）を返す"""
        pass

    @abstractmethod
    def get_lm_head(self) -> nn.Module:
        """言語モデル出力層（線形層）を返す"""
        pass

    def get_hook_target(self, layer_idx: int, hook_point: str) -> nn.Module:
        """
        概念名フックポイントから該当モジュールを取得
        hook_point:
          - 'layer': レイヤー全体（入力はpre_attn_resid、出力はpost_mlp_resid）
          - 'attn': Self-Attention モジュール
          - 'mlp': MLP モジュール
        """
        layer = self.get_layers()[layer_idx]
        if hook_point == "layer":
            return layer
        elif hook_point == "attn":
            return self.get_attn_module(layer_idx)
        elif hook_point == "mlp":
            return self.get_mlp_module(layer_idx)
        else:
            raise ValueError(f"Unknown hook_point: {hook_point}")


class LlamaFamilyAdapter(ModelAdapter):
    """Llama 3 / Mistral / Qwen 2.5 共通アダプター（HuggingFace Llama/Qwen2/MistralForCausalLM）"""

    def __init__(self, model: nn.Module):
        super().__init__(model)
        # model.model または model が基盤トランスフォーマー
        if hasattr(model, "model"):
            self.transformer = model.model
        elif hasattr(model, "transformer"):
            self.transformer = model.transformer
        else:
            self.transformer = model

    def get_layers(self) -> nn.ModuleList:
        return self.transformer.layers

    def get_attn_module(self, layer_idx: int) -> nn.Module:
        return self.get_layers()[layer_idx].self_attn

    def get_mlp_module(self, layer_idx: int) -> nn.Module:
        return self.get_layers()[layer_idx].mlp

    def get_final_norm(self) -> nn.Module:
        return self.transformer.norm

    def get_lm_head(self) -> nn.Module:
        return self.model.lm_head


class Gemma2Adapter(ModelAdapter):
    """Gemma 2 アダプター（Gemma2ForCausalLM）"""

    def __init__(self, model: nn.Module):
        super().__init__(model)
        if hasattr(model, "model"):
            self.transformer = model.model
        else:
            self.transformer = model

    def get_layers(self) -> nn.ModuleList:
        return self.transformer.layers

    def get_attn_module(self, layer_idx: int) -> nn.Module:
        return self.get_layers()[layer_idx].self_attn

    def get_mlp_module(self, layer_idx: int) -> nn.Module:
        return self.get_layers()[layer_idx].mlp

    def get_final_norm(self) -> nn.Module:
        return self.transformer.norm

    def get_lm_head(self) -> nn.Module:
        return self.model.lm_head


def get_model_adapter(model: nn.Module) -> ModelAdapter:
    """モデルクラスに応じて適切な ModelAdapter を返すファクトリ関数"""
    cls_name = model.__class__.__name__.lower()
    if "gemma2" in cls_name:
        return Gemma2Adapter(model)
    elif any(arch in cls_name for arch in ("llama", "qwen2", "mistral")):
        return LlamaFamilyAdapter(model)
    else:
        # フォールバックとして一般的な Transformer 構造を試みる
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return LlamaFamilyAdapter(model)
        raise NotImplementedError(f"No adapter available for model class {model.__class__.__name__}")
