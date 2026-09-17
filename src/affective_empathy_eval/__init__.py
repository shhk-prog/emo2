"""
Affective Empathy Evaluation Library (Behavioral / V1 / V2 / V3 統一基盤)
"""

__version__ = "0.3.0"

from affective_empathy_eval.controls import (
    generate_neutral_control,
    generate_scrambled_control,
)
from affective_empathy_eval.data import (
    load_aipsy_csv,
    load_emobank_csv,
)
from affective_empathy_eval.diagnostics import (
    MultivariateActivationDiagnostics,
)
from affective_empathy_eval.evaluation import (
    Evaluator,
)
from affective_empathy_eval.extraction import (
    extract_activations_batch,
)
from affective_empathy_eval.geometry import (
    GramSchmidtOrthogonalizer,
    compute_cosine_similarity,
    remove_projection_subspace,
)
from affective_empathy_eval.intervention import (
    ActivationPatcher,
    HookManager,
    RepresentationSteering,
)
from affective_empathy_eval.interventions import (
    InterchangeHook,
    InterventionController,
    SubspaceDeflectionHook,
)
from affective_empathy_eval.likelihood import (
    build_sequence_candidate_inputs,
    compute_sequence_likelihoods_for_candidates,
    score_joint_grid_distribution,
)
from affective_empathy_eval.manifests import (
    create_run_manifest,
)
from affective_empathy_eval.metrics import (
    compute_directional_alignment,
)
from affective_empathy_eval.optimal_transport import (
    compute_1d_wasserstein,
    compute_joint_ot_2d,
    compute_joint_ot_recovery,
    compute_marginal_wasserstein_sum,
)
from affective_empathy_eval.probing import (
    CrossDecoder,
    ProbeResult,
    RidgeProbe,
)
from affective_empathy_eval.schemas import (
    AffectiveReportSchema,
    LikelihoodResultSchema,
)
from affective_empathy_eval.splits import (
    split_discovery_confirmation,
)
from affective_empathy_eval.statistics import (
    apply_benjamini_hochberg,
    compute_bootstrap_ci,
    compute_correlation_with_ci,
    compute_d_z,
)

__all__ = [
    "compute_sequence_likelihoods_for_candidates",
    "build_sequence_candidate_inputs",
    "score_joint_grid_distribution",
    "GramSchmidtOrthogonalizer",
    "compute_cosine_similarity",
    "remove_projection_subspace",
    "InterventionController",
    "SubspaceDeflectionHook",
    "InterchangeHook",
    "HookManager",
    "ActivationPatcher",
    "RepresentationSteering",
    "compute_d_z",
    "compute_bootstrap_ci",
    "apply_benjamini_hochberg",
    "compute_correlation_with_ci",
    "RidgeProbe",
    "CrossDecoder",
    "ProbeResult",
    "compute_joint_ot_2d",
    "compute_marginal_wasserstein_sum",
    "compute_1d_wasserstein",
    "compute_joint_ot_recovery",
    "MultivariateActivationDiagnostics",
    "extract_activations_batch",
    "load_emobank_csv",
    "load_aipsy_csv",
    "generate_scrambled_control",
    "generate_neutral_control",
    "split_discovery_confirmation",
    "compute_directional_alignment",
    "AffectiveReportSchema",
    "LikelihoodResultSchema",
    "create_run_manifest",
    "Evaluator",
]
