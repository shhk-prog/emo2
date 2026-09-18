"""
v3/src/model_utils.py

Model abstraction, safe hook closures, and token-level alignment utilities
supporting both Qwen2.5 and Llama-3/3.2 architectures.
"""

import torch
import numpy as np

def get_model_layers(model):
    """
    Returns the module list of transformer layers for Qwen2 or LLaMA architecture.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "layers"):
        return model.layers
    else:
        raise AttributeError(f"Unable to locate transformer layers in model of type {type(model)}")

def get_component_module(model, layer_idx, comp="mlp"):
    """
    Returns the target submodule for activation patching or ablation:
      - 'mlp': MLP submodule (prior to residual addition)
      - 'attn': self_attn submodule (projected attention output prior to residual addition)
      - 'resid': TransformerBlock layer itself (block output post residual addition)
    """
    layers = get_model_layers(model)
    layer = layers[layer_idx]
    if comp == "mlp":
        return layer.mlp
    elif comp in ["attn", "self_attn"]:
        return layer.self_attn
    elif comp in ["resid", "layer"]:
        return layer
    else:
        raise ValueError(f"Unknown component: {comp}. Must be one of ['mlp', 'attn', 'resid'].")

def get_patch_hook(source_tensor, target_pos):
    """
    Robust hook closure for patching hidden states at target_pos.
    Handles arbitrary batch sizes (e.g. batched 81 candidate evaluation),
    and safe for both Tensor and tuple outputs.
    """
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            if target_pos < h.shape[1]:
                h[:, target_pos, :] = source_tensor.to(device=h.device, dtype=h.dtype)
            return (h, *output[1:])
        elif isinstance(output, torch.Tensor):
            if target_pos < output.shape[1]:
                output[:, target_pos, :] = source_tensor.to(device=output.device, dtype=output.dtype)
            return output
        else:
            return output
    return hook

def get_direction_ablation_hook(direction_unit_vector, target_pos):
    """
    Hook closure that removes the projection onto direction_unit_vector across all batch items:
      h_pos <- h_pos - (h_pos . v) * v
    where v is a unit vector (||v||_2 = 1.0).
    """
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            if target_pos < h.shape[1]:
                v = torch.as_tensor(direction_unit_vector, device=h.device, dtype=h.dtype)
                v = v / (torch.norm(v) + 1e-12)
                h_pos = h[:, target_pos, :]
                proj = torch.matmul(h_pos, v)
                h[:, target_pos, :] = h_pos - torch.outer(proj, v)
            return (h, *output[1:])
        elif isinstance(output, torch.Tensor):
            if target_pos < output.shape[1]:
                v = torch.as_tensor(direction_unit_vector, device=output.device, dtype=output.dtype)
                v = v / (torch.norm(v) + 1e-12)
                h_pos = output[:, target_pos, :]
                proj = torch.matmul(h_pos, v)
                output[:, target_pos, :] = h_pos - torch.outer(proj, v)
            return output
        else:
            return output
    return hook

def get_replacement_hook(replacement_tensor, target_pos):
    """
    Hook closure that replaces activation with a static vector across all batch items.
    """
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            if target_pos < h.shape[1]:
                rep = torch.as_tensor(replacement_tensor, device=h.device, dtype=h.dtype)
                h[:, target_pos, :] = rep
            return (h, *output[1:])
        elif isinstance(output, torch.Tensor):
            if target_pos < output.shape[1]:
                rep = torch.as_tensor(replacement_tensor, device=output.device, dtype=output.dtype)
                output[:, target_pos, :] = rep
            return output
        else:
            return output
    return hook

def format_base_prompt(text):
    """
    Formats the evaluation prompt asking for JSON report.
    """
    return f"Read the following text and report your affective state.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."

def build_generation_prefix_inputs(tokenizer, prompt_text, prefix_str='{"valence": ', device="cpu"):
    """
    Constructs prompt inputs teacher-forced with prefix_str, ensuring strict token-level alignment.
    
    Verifies via assertion that the final token IDs strictly match the encoded prefix_str.
    Returns:
        inputs (BatchEncoding): Tokenized inputs on the target device
        target_pos (int): Index of the final token of the prefix (position where next token is predicted)
        prefix_ids (list[int]): Expected prefix token IDs
    """
    # 1. Encode prefix alone to get expected token IDs
    # Note: We encode prefix directly; if tokenizer prepends BOS, we strip it
    prefix_tokens = tokenizer(prefix_str, add_special_tokens=False).input_ids
    
    # 2. Combine prompt and prefix
    full_text = prompt_text + prefix_str
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    
    seq_len = inputs.input_ids.shape[1]
    target_pos = seq_len - 1
    
    # 3. Token-level assertion: verify that the tail of input_ids matches prefix_tokens
    actual_tail = inputs.input_ids[0, target_pos - len(prefix_tokens) + 1 : target_pos + 1].tolist()
    assert actual_tail == prefix_tokens, (
        f"Token-level prefix mismatch!\n"
        f"Expected prefix IDs: {prefix_tokens}\n"
        f"Actual tail IDs:     {actual_tail}\n"
        f"Decoded tail: '{tokenizer.decode(actual_tail)}' vs expected '{prefix_str}'"
    )
    
    return inputs, target_pos, prefix_tokens
