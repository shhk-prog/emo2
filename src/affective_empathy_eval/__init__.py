"""
Affective Empathy Evaluation Library (Behavioral / V1 / V2 / V3 統一基盤)
"""

__version__ = "0.3.0"

# Controls
from affective_empathy_eval.controls import (
    calculate_empirical_p_value,
    filter_control_subsets,
    generate_neutral_control,
    generate_scrambled_control,
    generate_shuffled_control_dataset,
    is_lexical_emotion,
)

# Data
from affective_empathy_eval.data import (
    load_aipsy_affect,
    load_aipsy_csv,
    load_emobank,
    load_emobank_csv,
    load_v3_matched_pair_table,
    resolve_matched_neutral_text,
    scale_vad,
    split_aipsy_affect,
    stratify_stimuli,
    stratified_causal_subset,
    v3_stimulus_covariate,
)

# Diagnostics
from affective_empathy_eval.diagnostics import (
    MultivariateActivationDiagnostics,
)

# Evaluation
from affective_empathy_eval.evaluation import (
    Evaluator,
    InterventionEvaluator,
)

# Extraction
from affective_empathy_eval.extraction import (
    MockRepresentationExtractor,
    PyTorchRepresentationExtractor,
    RepresentationExtractor,
    extract_activations_batch,
)

# Geometry
from affective_empathy_eval.geometry import (
    compute_center_of_mass,
    compute_dissociation_metrics,
    compute_layer_dissociation,
    compute_peak_depth,
    compute_relative_depth,
    compute_rsa_correlation,
    eval_held_out_cross_decoding,
    eval_held_out_procrustes,
    get_block_hidden_state,
    train_and_eval_held_out_probe,
)


# Intervention
from affective_empathy_eval.intervention import (
    ActivationPatcher,
    HookManager,
    PatchPair,
    RepresentationAblator,
    RepresentationSteering,
    SteeringController,
)

# Interventions (Subspace & Hooks)
from affective_empathy_eval.interventions import (
    apply_centered_projection_removal_1d,
    apply_centered_projection_removal_subspace,
    compute_causal_leverage,
    compute_orthonormal_subspace,
    estimate_interventional_slope,
    extract_conditional_directions,
    generate_control_directions,
)

# Likelihood & Candidates
from affective_empathy_eval.likelihood import (
    build_va_candidates,
    build_vad_candidates,
    compute_distribution_metrics,
    compute_emd_recovery_ratio,
    compute_emd_va,
    compute_expected_va,
    compute_marginal_distributions,
    compute_sequence_likelihoods_for_candidates,
    evaluate_expected_va_from_prompt,
    evaluate_expected_vad_from_prompt,
    get_euclidean_ground_cost_matrix,
    prepare_joint_sequence_with_boundary,
    resolve_joint_stage_index,
)

# Manifests
from affective_empathy_eval.manifests import (
    ExtractionManifest,
    ManifestManager,
)

# Metrics
from affective_empathy_eval.metrics import (
    calculate_anchor_direction_alignment,
    calculate_euclidean_recovery,
    calculate_post_distance,
    calculate_reactivity_vector,
    calculate_stimulus_gain,
    compute_rsa_similarity,
)

# Optimal Transport
from affective_empathy_eval.optimal_transport import (
    compute_1d_wasserstein,
    compute_joint_ot_2d,
    compute_joint_ot_recovery,
    compute_marginal_wasserstein_sum,
)

# Probing
from affective_empathy_eval.probing import (
    FixedSplitProber,
    LayerProber,
)

# Affect Directions
from affective_empathy_eval.affect_directions import (
    AIPSY_EXPECTED_DIRECTION,
    get_expected_sign,
)

# Prompts
from affective_empathy_eval.prompts import (
    TaskType,
    build_prompt,
    encode_prompt_canonical,
    find_semantic_anchors,
    get_generation_stage_tokens,
    validate_stage_index_invariance,
)

# Schemas
from affective_empathy_eval.schemas import (
    AffectiveState,
    parse_affective_state,
)

# Splits
from affective_empathy_eval.splits import (
    create_dev_test_split,
    get_grouped_kfold_splits,
)

# Statistics
from affective_empathy_eval.statistics import (
    apply_benjamini_hochberg,
    apply_fdr_correction,
    cluster_based_permutation_test,
    compute_bivariate_bootstrap_ci,
    compute_bootstrap_ci,
    compute_correlation_with_ci,
    compute_d_z,
    compute_paired_cohen_dz,
    fit_sample_level_lmm,
    generate_derangement,
    paired_family_comparison,
)

__all__ = [
    # Controls
    "is_lexical_emotion",
    "filter_control_subsets",
    "generate_shuffled_control_dataset",
    "generate_scrambled_control",
    "generate_neutral_control",
    "calculate_empirical_p_value",
    # Data
    "scale_vad",
    "load_emobank",
    "load_emobank_csv",
    "load_v3_matched_pair_table",
    "resolve_matched_neutral_text",
    "stratify_stimuli",
    "load_aipsy_affect",
    "load_aipsy_csv",
    "split_aipsy_affect",
    "stratified_causal_subset",
    "v3_stimulus_covariate",
    # Diagnostics
    "MultivariateActivationDiagnostics",
    # Evaluation
    "InterventionEvaluator",
    "Evaluator",
    # Extraction
    "MockRepresentationExtractor",
    "PyTorchRepresentationExtractor",
    "RepresentationExtractor",
    "extract_activations_batch",
    # Geometry
    "train_and_eval_held_out_probe",
    "eval_held_out_cross_decoding",
    "eval_held_out_procrustes",
    "compute_rsa_correlation",
    "compute_relative_depth",
    "compute_center_of_mass",
    "compute_peak_depth",
    "compute_dissociation_metrics",
    "compute_layer_dissociation",
    # Intervention
    "PatchPair",
    "ActivationPatcher",
    "RepresentationAblator",
    "SteeringController",
    "HookManager",
    "RepresentationSteering",
    # Interventions
    "extract_conditional_directions",
    "compute_orthonormal_subspace",
    "generate_control_directions",
    "estimate_interventional_slope",
    "compute_causal_leverage",
    "apply_centered_projection_removal_1d",
    "apply_centered_projection_removal_subspace",
    # Likelihood
    "build_va_candidates",
    "build_vad_candidates",
    "compute_expected_va",
    "compute_marginal_distributions",
    "get_euclidean_ground_cost_matrix",
    "compute_emd_va",
    "compute_emd_recovery_ratio",
    "compute_distribution_metrics",
    "prepare_joint_sequence_with_boundary",
    "resolve_joint_stage_index",
    "compute_sequence_likelihoods_for_candidates",
    "evaluate_expected_va_from_prompt",
    "evaluate_expected_vad_from_prompt",
    # Manifests
    "ExtractionManifest",
    "ManifestManager",
    # Metrics
    "calculate_reactivity_vector",
    "calculate_anchor_direction_alignment",
    "calculate_post_distance",
    "calculate_stimulus_gain",
    "compute_rsa_similarity",
    "calculate_euclidean_recovery",
    # Optimal Transport
    "compute_joint_ot_2d",
    "compute_marginal_wasserstein_sum",
    "compute_1d_wasserstein",
    "compute_joint_ot_recovery",
    # Probing
    "LayerProber",
    "FixedSplitProber",
    # Prompts
    "TaskType",
    "build_prompt",
    "encode_prompt_canonical",
    "find_semantic_anchors",
    "get_generation_stage_tokens",
    "validate_stage_index_invariance",
    # Affect Directions
    "AIPSY_EXPECTED_DIRECTION",
    "get_expected_sign",
    # Schemas
    "AffectiveState",
    "parse_affective_state",
    # Splits
    "create_dev_test_split",
    "get_grouped_kfold_splits",
    # Statistics
    "fit_sample_level_lmm",
    "compute_bootstrap_ci",
    "compute_bivariate_bootstrap_ci",
    "paired_family_comparison",
    "apply_fdr_correction",
    "apply_benjamini_hochberg",
    "cluster_based_permutation_test",
    "compute_paired_cohen_dz",
    "compute_d_z",
    "generate_derangement",
    "compute_correlation_with_ci",
]
