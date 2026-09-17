import pytest
import numpy as np
from affective_empathy_eval.intervention import (
    ActivationPatcher, RepresentationAblator, SteeringController
)

def test_activation_patcher():
    patcher = ActivationPatcher(target_layer=12)
    target = np.array([1.0, 1.0])
    source = np.array([0.0, 0.0])
    
    patched = patcher.patch_activation(target, source, patch_weight=0.5)
    assert np.allclose(patched, np.array([0.5, 0.5]))

def test_representation_ablator():
    ablator_zero = RepresentationAblator(ablation_type="zero")
    ablated = ablator_zero.ablate(np.array([1.0, 2.0]))
    assert np.allclose(ablated, np.array([0.0, 0.0]))

def test_steering_controller():
    high_vecs = np.array([[2.0, 2.0], [2.0, 2.0]])
    low_vecs = np.array([[0.0, 0.0], [0.0, 0.0]])
    steering = SteeringController.compute_direction_from_contrast(high_vecs, low_vecs)
    
    target = np.array([1.0, 1.0])
    steered = steering.apply_steering(target, alpha=1.0)
    assert steered[0] > target[0]
