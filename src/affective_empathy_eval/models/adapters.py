from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """
    アーキテクチャ間の差異（Qwen, Llama, Gemma, OLMo, Mistral）を吸収する共通アダプター
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
        layers = self.get_layers()
        if layer_idx < 0 or layer_idx >= len(layers):
            raise IndexError(f"Layer index {layer_idx} out of range [0, {len(layers) - 1}]")

        layer = layers[layer_idx]
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
        if hasattr(model, "model"):
            self.transformer = model.model
        elif hasattr(model, "transformer"):
            self.transformer = model.transformer
        else:
            self.transformer = model

    def get_layers(self) -> nn.ModuleList:
        if hasattr(self.transformer, "layers"):
            return self.transformer.layers
        elif hasattr(self.transformer, "h"):
            return self.transformer.h
        raise AttributeError("Cannot locate layers in LlamaFamily model.")

    def get_attn_module(self, layer_idx: int) -> nn.Module:
        layer = self.get_layers()[layer_idx]
        if hasattr(layer, "self_attn"):
            return layer.self_attn
        elif hasattr(layer, "attn"):
            return layer.attn
        raise AttributeError(f"Cannot locate self_attn in layer {layer_idx}")

    def get_mlp_module(self, layer_idx: int) -> nn.Module:
        layer = self.get_layers()[layer_idx]
        if hasattr(layer, "mlp"):
            return layer.mlp
        elif hasattr(layer, "feed_forward"):
            return layer.feed_forward
        raise AttributeError(f"Cannot locate mlp in layer {layer_idx}")

    def get_final_norm(self) -> nn.Module:
        if hasattr(self.transformer, "norm"):
            return self.transformer.norm
        elif hasattr(self.transformer, "final_layernorm"):
            return self.transformer.final_layernorm
        elif hasattr(self.transformer, "ln_f"):
            return self.transformer.ln_f
        raise AttributeError("Cannot locate final norm in LlamaFamily model.")

    def get_lm_head(self) -> nn.Module:
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        elif hasattr(self.transformer, "lm_head"):
            return self.transformer.lm_head
        raise AttributeError("Cannot locate lm_head in LlamaFamily model.")


class GemmaAdapter(ModelAdapter):
    """Gemma 2 & Gemma 3 アダプター（Gemma2ForCausalLM, Gemma3ForCausalLM）"""

    def __init__(self, model: nn.Module):
        super().__init__(model)
        if hasattr(model, "model"):
            self.transformer = model.model
        else:
            self.transformer = model

    def get_layers(self) -> nn.ModuleList:
        return self.transformer.layers

    def get_attn_module(self, layer_idx: int) -> nn.Module:
        layer = self.get_layers()[layer_idx]
        return getattr(layer, "self_attn", getattr(layer, "attn", None))

    def get_mlp_module(self, layer_idx: int) -> nn.Module:
        layer = self.get_layers()[layer_idx]
        return getattr(layer, "mlp", getattr(layer, "feed_forward", None))

    def get_final_norm(self) -> nn.Module:
        return getattr(self.transformer, "norm", getattr(self.transformer, "final_layernorm", None))

    def get_lm_head(self) -> nn.Module:
        return self.model.lm_head


class Olmo2Adapter(ModelAdapter):
    """OLMo 2 アダプター（OLMo2ForCausalLM / OLMoForCausalLM）"""

    def __init__(self, model: nn.Module):
        super().__init__(model)
        if hasattr(model, "model"):
            self.transformer = model.model
        elif hasattr(model, "transformer"):
            self.transformer = model.transformer
        else:
            self.transformer = model

    def get_layers(self) -> nn.ModuleList:
        if hasattr(self.transformer, "layers"):
            return self.transformer.layers
        elif hasattr(self.transformer, "blocks"):
            return self.transformer.blocks
        raise AttributeError("Cannot locate layers in OLMo model.")

    def get_attn_module(self, layer_idx: int) -> nn.Module:
        layer = self.get_layers()[layer_idx]
        if hasattr(layer, "self_attn"):
            return layer.self_attn
        elif hasattr(layer, "att_proj"):
            return layer.att_proj
        elif hasattr(layer, "attn"):
            return layer.attn
        raise AttributeError(f"Cannot locate attention module in OLMo layer {layer_idx}")

    def get_mlp_module(self, layer_idx: int) -> nn.Module:
        layer = self.get_layers()[layer_idx]
        if hasattr(layer, "mlp"):
            return layer.mlp
        elif hasattr(layer, "ff_proj"):
            return layer.ff_proj
        raise AttributeError(f"Cannot locate MLP module in OLMo layer {layer_idx}")

    def get_final_norm(self) -> nn.Module:
        if hasattr(self.transformer, "norm"):
            return self.transformer.norm
        elif hasattr(self.transformer, "ln_f"):
            return self.transformer.ln_f
        raise AttributeError("Cannot locate final norm in OLMo model.")

    def get_lm_head(self) -> nn.Module:
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        elif hasattr(self.transformer, "ff_out"):
            return self.transformer.ff_out
        raise AttributeError("Cannot locate lm_head in OLMo model.")


def get_model_adapter(model: nn.Module, adapter_name: Any = None) -> ModelAdapter:
    """
    モデルインスタンスおよびオプショナルな adapter_name または ModelFamilyConfig から適切な ModelAdapter を生成する。
    """
    if adapter_name is not None:
        if hasattr(adapter_name, "adapter") and adapter_name.adapter:
            adapter_key = str(adapter_name.adapter).lower()
        elif hasattr(adapter_name, "family_id") and adapter_name.family_id:
            adapter_key = str(adapter_name.family_id).lower()
        else:
            adapter_key = str(adapter_name).lower()

        if "gemma" in adapter_key:
            return GemmaAdapter(model)
        elif "olmo" in adapter_key:
            return Olmo2Adapter(model)
        elif any(k in adapter_key for k in ("llama", "qwen", "mistral")):
            return LlamaFamilyAdapter(model)

    cls_name = model.__class__.__name__.lower()
    if "gemma" in cls_name:
        return GemmaAdapter(model)
    elif "olmo" in cls_name:
        return Olmo2Adapter(model)
    elif any(arch in cls_name for arch in ("llama", "qwen", "mistral")):
        return LlamaFamilyAdapter(model)
    else:
        # フォールバックとして構造検査
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return LlamaFamilyAdapter(model)
        if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
            return LlamaFamilyAdapter(model)
        raise NotImplementedError(f"No adapter available for model class {model.__class__.__name__}")
