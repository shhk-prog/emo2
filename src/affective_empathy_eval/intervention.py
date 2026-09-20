import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass

@dataclass
class PatchPair:
    """Represents a pair of stimuli for Activation Patching experiments."""
    source_id: str
    target_id: str
    source_expected_v: float
    source_expected_a: float
    target_expected_v: float
    target_expected_a: float
    metadata: Dict[str, Any] = None


class ActivationPatcher:
    """Implements Activation Patching (Clean, Corrupted, Patched runs)."""
    def __init__(self, target_layer: int, position_type: str = "stimulus_last_token"):
        self.target_layer = target_layer
        self.position_type = position_type

    def patch_activation(
        self, target_vector: np.ndarray, source_vector: np.ndarray, patch_weight: float = 1.0
    ) -> np.ndarray:
        """Interpolates target vector towards source vector by patch_weight (0=no patch, 1=full patch)."""
        return (1.0 - patch_weight) * target_vector + patch_weight * source_vector

    @staticmethod
    def calculate_recovery(clean_metric: float, corrupted_metric: float, patched_metric: float, eps: float = 1e-5) -> float:
        """
        Calculates the recovery score for a patching intervention.
        Recovery = (Patched - Corrupted) / (Clean - Corrupted)
        """
        denominator = clean_metric - corrupted_metric
        if abs(denominator) < eps:
            return np.nan
        return float((patched_metric - corrupted_metric) / denominator)


class RepresentationAblator:
    """Implements Ablation interventions, strictly focusing on Neutral replacement as main analysis."""
    def __init__(self, ablation_type: str = "neutral"):
        assert ablation_type in ["zero", "mean", "neutral"]
        self.ablation_type = ablation_type

    def ablate(self, target_vector: np.ndarray, reference_vector: Optional[np.ndarray] = None) -> np.ndarray:
        """Applies ablation."""
        if self.ablation_type == "zero":
            return np.zeros_like(target_vector)
        elif self.ablation_type in ["mean", "neutral"]:
            if reference_vector is None:
                raise ValueError(f"reference_vector must be provided for {self.ablation_type} ablation.")
            return reference_vector.copy().astype(target_vector.dtype)
        return target_vector


class SteeringController:
    """Computes and applies affective steering vectors to model representations."""
    def __init__(self, steering_direction: np.ndarray):
        norm = np.linalg.norm(steering_direction)
        self.steering_direction = steering_direction / (norm + 1e-8)

    @classmethod
    def compute_direction_from_contrast(
        cls, high_affect_vectors: np.ndarray, low_affect_vectors: np.ndarray
    ) -> 'SteeringController':
        """Computes mean difference vector (high - low) as steering direction using training dev set."""
        mean_high = np.mean(high_affect_vectors, axis=0)
        mean_low = np.mean(low_affect_vectors, axis=0)
        direction = mean_high - mean_low
        return cls(steering_direction=direction)

    def apply_steering(self, target_vector: np.ndarray, alpha: float) -> np.ndarray:
        return target_vector + (alpha * self.steering_direction).astype(target_vector.dtype)


class PyTorchActivationPatcher:
    """
    Context manager to apply Activation Patching, Ablation, or Steering to a HuggingFace model
    during generation using forward hooks.
    """
    def __init__(self, model, target_layer: int, source_tensor=None, patch_weight: float = 1.0, position: int = -1, intervention_type: str = "patch"):
        self.model = model
        self.target_layer = target_layer
        self.source_tensor = source_tensor
        self.patch_weight = patch_weight
        self.position = position
        self.intervention_type = intervention_type
        self.handle = None

    def __enter__(self):
        transformer_layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            transformer_layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            transformer_layers = self.model.transformer.h
            
        if transformer_layers is None or self.target_layer >= len(transformer_layers):
            raise ValueError(f"Invalid target layer {self.target_layer} for model architecture.")

        def hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            if hidden_states.shape[1] > 1:
                import torch
                original = hidden_states[:, self.position, :]

                if self.intervention_type in ("zero", "ablate_zero"):
                    patched = torch.zeros_like(original)
                elif self.intervention_type == "mean" and self.source_tensor is None:
                    patched = torch.zeros_like(original)
                else:
                    if self.source_tensor is None:
                        raise ValueError(f"source_tensor cannot be None for intervention_type '{self.intervention_type}'")
                    
                    if isinstance(self.source_tensor, torch.Tensor):
                        src = self.source_tensor.to(device=hidden_states.device, dtype=hidden_states.dtype)
                    else:
                        src = torch.tensor(self.source_tensor, device=hidden_states.device, dtype=hidden_states.dtype)
                    
                    if src.dim() == 1:
                        src = src.unsqueeze(0).expand(hidden_states.shape[0], -1)
                    elif src.dim() == 2 and src.shape[0] != hidden_states.shape[0]:
                        if src.shape[0] == 1:
                            src = src.expand(hidden_states.shape[0], -1)
                        else:
                            raise ValueError(f"source_tensor batch size {src.shape[0]} does not match hidden_states batch size {hidden_states.shape[0]}")
                            
                    if self.intervention_type == "patch":
                        patched = (1.0 - self.patch_weight) * original + self.patch_weight * src
                    elif self.intervention_type == "add":
                        patched = original + (self.patch_weight * src)
                    elif self.intervention_type in ("replace", "mean"):
                        patched = src
                    else:
                        raise ValueError(f"Unknown intervention_type: {self.intervention_type}")
                
                new_hidden = hidden_states.clone()
                new_hidden[:, self.position, :] = patched
                
                if isinstance(output, tuple):
                    return (new_hidden,) + output[1:]
                return new_hidden
                
            return output

        self.handle = transformer_layers[self.target_layer].register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# Backward-compatibility aliases
HookManager = ActivationPatcher
RepresentationSteering = SteeringController
