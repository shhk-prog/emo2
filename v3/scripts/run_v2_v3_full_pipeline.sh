#!/usr/bin/env bash
set -euo pipefail

# Ensure unbuffered python output for real-time streaming
export PYTHONUNBUFFERED=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure virtual environment is activated
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -d ".venv" ]; then
        echo "Activating virtual environment (.venv)..."
        source .venv/bin/activate
    else
        echo "Error: Virtual environment (.venv) not found in $PROJECT_ROOT."
        exit 1
    fi
fi

# Ensure base library affective_empathy_eval in src/ is in PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

DEVICE="cuda"
DRY_RUN=false
TARGET_STEP=""
FROM_STEP=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            DEVICE="cpu"
            shift
            ;;
        --step)
            TARGET_STEP="$2"
            shift 2
            ;;
        --from-step)
            FROM_STEP="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--device <cuda|cpu>] [--dry-run] [--step <1-7>] [--from-step <1-7>]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

COMMON_ARGS="--device $DEVICE"
if [ "$DRY_RUN" = true ]; then
    COMMON_ARGS="--dry-run --device cpu"
    echo ">>> Running Pipeline in DRY-RUN mode <<<"
fi

LOG_DIR="v3/results/logs/pipeline"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FULL_LOG="$LOG_DIR/full_pipeline_${TIMESTAMP}.log"

echo "========================================================================"
echo " Starting V2 & V3 Pipeline: Steps 1 to 7"
echo " Mode: $( [ "$DRY_RUN" = true ] && echo "DRY-RUN" || echo "LIVE ($DEVICE)" )"
echo " Working Directory: $(pwd)"
echo " Per-Step Logs:     $LOG_DIR/stepX_<name>_${TIMESTAMP}.log"
echo " Full Combined Log: $FULL_LOG"
echo "========================================================================"

run_step() {
    local step_num=$1
    local step_slug=$2
    local step_title=$3
    local step_cmd=$4

    if [ -n "$TARGET_STEP" ] && [ "$TARGET_STEP" -ne "$step_num" ]; then
        return 0
    fi
    if [ "$step_num" -lt "$FROM_STEP" ]; then
        return 0
    fi

    local step_log="$LOG_DIR/step${step_num}_${step_slug}_${TIMESTAMP}.log"
    local latest_link="$LOG_DIR/latest_step${step_num}.log"

    echo ""
    echo "========================================================================"
    echo " [Step $step_num/7] $step_title"
    echo " Started at:  $(date '+%Y-%m-%d %H:%M:%S')"
    echo " Command:     $step_cmd"
    echo " Step Log:    $step_log"
    echo "========================================================================"

    local start_time
    start_time=$(date +%s)

    # Execute and stream in real-time to terminal, step-specific log, and combined log
    (
        eval "$step_cmd"
    ) 2>&1 | tee -a "$step_log" | tee -a "$FULL_LOG"

    # Maintain a symlink to the latest step log
    ln -sf "$(basename "$step_log")" "$latest_link"

    local end_time
    end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local minutes=$((elapsed / 60))
    local seconds=$((elapsed % 60))

    echo "------------------------------------------------------------------------"
    echo " ✓ Step $step_num Completed successfully in ${minutes}m ${seconds}s"
    echo " Log saved to: $step_log"
    echo "------------------------------------------------------------------------"
}

# --- Step 1: Base Testing ---
run_step 1 "tests" \
    "Common Base Library Tests (affective_empathy_eval)" \
    "pytest tests/"

# --- Step 2: V2 Cross-decoding & Geometry (4 Families) ---
run_step 2 "v2_cross_decoding" \
    "V2-RQ1 & RQ2: Cross-decoding and Geometry Analysis" \
    "python v2/scripts/run_v2_2x2_cross_decoding.py $COMMON_ARGS"

# --- Step 3: V2 Causal Map & Peak Dissociation (4 Families) ---
run_step 3 "v2_causal_map" \
    "V2-RQ3: Causal Map & Peak Dissociation Analysis" \
    "python v2/scripts/run_v2_2x2_causal_map.py $COMMON_ARGS"

# --- Step 4: V2 Distribution Recovery Patching ---
run_step 4 "v2_recovery_patching" \
    "V2-RQ4: Distribution Recovery Patching (EMD_VA)" \
    "python v2/scripts/run_v2_recovery_patching.py $COMMON_ARGS"

# --- Step 5: V3 State Induction & Go/No-Go Gate ---
run_step 5 "v3_state_induction" \
    "V3-RQ1: Internal Affective State Induction & Go/No-Go Gate" \
    "python v3/scripts/run_v3_state_induction.py $COMMON_ARGS"

# --- Step 6: V3 Spatiotemporal 4-Map & Path Mediation ---
run_step 6 "v3_spatiotemporal_mediation" \
    "V3-RQ2 & RQ3: Spatiotemporal 4-Map & Path Mediation (Qwen)" \
    "python v3/scripts/run_v3_spatiotemporal_maps.py $COMMON_ARGS && python v3/scripts/run_v3_path_mediation.py $COMMON_ARGS"

# --- Step 7: V3 Confirmatory Replication across 3 Models ---
run_step 7 "v3_confirmatory_replication" \
    "V3: Confirmatory Replication across Llama, Gemma, Mistral" \
    "python v3/scripts/run_v3_confirmatory_replication.py $COMMON_ARGS"

echo ""
echo "========================================================================"
echo " ✓ All V2 & V3 Pipeline Steps Completed Successfully!"
echo " Finished at:       $(date '+%Y-%m-%d %H:%M:%S')"
echo " Full Combined Log: $FULL_LOG"
echo " Results stored in: v2/results/ and v3/results/"
echo "========================================================================"
