#!/usr/bin/env bash
set -e

# ==============================================================================
# V1 Phase C: Modular Causal Intervention Runner
# Evaluates:
#   - E3 & E4: Causal Patching (Shared Causal Map & Difference Vector Interchangeability)
#   - E6: Targeted Ablation & Double Dissociation (LMM Interaction Test)
#
# Allows running full suite or splitting into individual jobs by model, task, or job ID.
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure virtual environment is activated
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -d ".venv" ]; then
        echo "Activating virtual environment (.venv)..."
        source .venv/bin/activate
    else
        echo "Error: Virtual environment (.venv) not found."
        exit 1
    fi
fi

export PYTHONPATH="$PROJECT_ROOT/v1/src:$PROJECT_ROOT/src:${PYTHONPATH:-}"

# Model definitions: "model_id:prefix:is_instruct:reader_layer:self_layer"
MODELS=(
    "Qwen/Qwen2.5-1.5B:qwen2.5_1.5b_base:false:8:20"
    "Qwen/Qwen2.5-1.5B-Instruct:qwen2.5_1.5b_instruct:true:8:20"
    "meta-llama/Llama-3.2-1B:llama3.2_1b_base:false:4:12"
    "meta-llama/Llama-3.2-1B-Instruct:llama3.2_1b_instruct:true:4:12"
    "google/gemma-2-2b:gemma2_2b_base:false:6:18"
    "google/gemma-2-2b-it:gemma2_2b_instruct:true:6:18"
    "mistralai/Mistral-7B-v0.1:mistral7b_v0.1_base:false:10:24"
    "mistralai/Mistral-7B-Instruct-v0.1:mistral7b_v0.1_instruct:true:10:24"
)

# Build job list
JOBS=()
JOB_COMMANDS=()
JOB_DESCS=()

IDX=1
for m in "${MODELS[@]}"; do
    IFS=":" read -r model_id prefix is_instruct r_layer s_layer <<< "$m"

    # Task 1: E3 & E4 Causal Patching
    job_id_patch="${prefix}_patching"
    JOBS+=("$job_id_patch")
    JOB_DESCS+=("[$IDX] $prefix: E3/E4 Causal Patching & Difference Transfer")
    
    cmd_patch="python v1/scripts/run_v1_phase_c_causal_patching.py --model-id \"$model_id\" --model-prefix \"$prefix\""
    JOB_COMMANDS+=("$cmd_patch")
    IDX=$((IDX + 1))

    # Task 2: E6 Targeted Ablation (Double Dissociation)
    job_id_abl="${prefix}_ablation"
    JOBS+=("$job_id_abl")
    JOB_DESCS+=("[$IDX] $prefix: E6 Double Dissociation Ablation (L$r_layer vs L$s_layer)")
    
    cmd_abl="python v1/scripts/run_v1_phase_c_targeted_ablation.py --model-id \"$model_id\" --model-prefix \"$prefix\" --reader-layer $r_layer --self-layer $s_layer"
    JOB_COMMANDS+=("$cmd_abl")
    IDX=$((IDX + 1))
done

# Help & Usage
usage() {
    echo "========================================================================"
    echo " V1 Phase C Causal Intervention Runner (Job-based Dispatcher)"
    echo "========================================================================"
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Execution Modes (Choose one):"
    echo "  --list, -l                   List all available jobs with their IDs"
    echo "  --job, -j <ID_OR_NAME>       Run a single job by number (1-${#JOBS[@]}) or name"
    echo "  --model, -m <PREFIX>         Run all Phase C jobs for a specific model prefix"
    echo "  --task, -t <patching|ablation> Run all models for a specific task"
    echo "  --all, -a                    Run ALL jobs for all models sequentially"
    echo ""
    echo "Tuning & Control Options:"
    echo "  --all-layers                 Run E3/E4 causal patching across ALL layers"
    echo "  --limit <N>                  Limit matched pairs (default: 0 = full 480 pairs)"
    echo "  --sub-batch-size <N>         Sub-batch size for 729-candidate eval (default: 81)"
    echo "  --force, -f                  Force re-run even if output results already exist"
    echo "  --dry-run                    Print commands without executing them"
    echo "  --help, -h                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --list"
    echo "  $0 --job 2                   # Run Job 2 (qwen2.5_1.5b_base_ablation)"
    echo "  $0 --job qwen2.5_1.5b_instruct_patching --all-layers"
    echo "  $0 --model llama3.2_1b_instruct"
    echo "  $0 --task ablation           # Run E6 Double Dissociation for all 8 models"
    echo "========================================================================"
    exit 0
}

# Parse options
MODE=""
TARGET_JOB=""
TARGET_MODEL=""
TARGET_TASK=""
ALL_LAYERS=false
LIMIT=0
DEVICE="cuda"
DRY_RUN=false
SUB_BATCH_SIZE=81
FORCE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list|-l)
            MODE="list"
            shift
            ;;
        --job|-j)
            MODE="job"
            TARGET_JOB="$2"
            shift 2
            ;;
        --model|-m)
            MODE="model"
            TARGET_MODEL="$2"
            shift 2
            ;;
        --task|-t)
            MODE="task"
            TARGET_TASK="$2"
            shift 2
            ;;
        --all|-a)
            MODE="all"
            shift
            ;;
        --all-layers)
            ALL_LAYERS=true
            shift
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --sub-batch-size)
            SUB_BATCH_SIZE="$2"
            shift 2
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [ -z "$MODE" ]; then
    if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
        MODE="job"
        TARGET_JOB="$SLURM_ARRAY_TASK_ID"
        echo "[Slurm Detection] Running as Slurm Array Task ID: $SLURM_ARRAY_TASK_ID"
    else
        echo "Error: No execution mode specified."
        usage
    fi
fi

# 1. List Jobs Mode
if [ "$MODE" = "list" ]; then
    echo "========================================================================"
    echo " Available Phase C Jobs (${#JOBS[@]} Total):"
    echo "========================================================================"
    for i in "${!JOBS[@]}"; do
        echo "${JOB_DESCS[$i]}  [id: ${JOBS[$i]}]"
    done
    echo "========================================================================"
    exit 0
fi

# Function to execute a job by index (0-based)
run_job_by_index() {
    local idx=$1
    local name="${JOBS[$idx]}"
    local base_cmd="${JOB_COMMANDS[$idx]}"

    # Check if results already exist for skipping
    local prefix=""
    local expected_file=""
    if [[ "$name" == *"_patching" ]]; then
        prefix="${name%_patching}"
        expected_file="v1/results/derived/v1_phase_c/${prefix}/phase_c_summary.md"
    elif [[ "$name" == *"_ablation" ]]; then
        prefix="${name%_ablation}"
        expected_file="v1/results/derived/v1_phase_c/${prefix}/e6_double_dissociation_summary.md"
    fi

    if [ "$FORCE" != true ] && [ -n "$expected_file" ] && [ -f "$expected_file" ]; then
        echo "========================================================================"
        echo "[SKIP] Job $((idx + 1)): $name is already completed."
        echo "Found output: $expected_file"
        echo "Skipping execution (use --force to re-run)."
        echo "========================================================================"
        return 0
    fi
    
    local cmd="$base_cmd --limit $LIMIT --device \"$DEVICE\" --sub-batch-size $SUB_BATCH_SIZE"
    if [ "$ALL_LAYERS" = true ] && [[ "$name" == *"patching"* ]]; then
        cmd="$cmd --all-layers"
    fi
    if [ "$FORCE" = true ]; then
        cmd="$cmd --force"
    fi

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_dir="v1/results/logs/phase_c"
    mkdir -p "$log_dir"
    local job_log="$log_dir/${name}_${timestamp}.log"

    echo ""
    echo "========================================================================"
    echo "Executing Job $((idx + 1)): $name"
    echo "Command: $cmd"
    echo "Log:     $job_log"
    echo "========================================================================"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Command skipped."
    else
        eval "$cmd" 2>&1 | tee "$job_log"
        echo "✓ Finished job: $name (Log saved to $job_log)"
    fi
}

# 2. Run Single Job Mode
if [ "$MODE" = "job" ]; then
    JOB_INDEX=-1
    if [[ "$TARGET_JOB" =~ ^[0-9]+$ ]]; then
        JOB_INDEX=$((TARGET_JOB - 1))
        if [ $JOB_INDEX -lt 0 ] || [ $JOB_INDEX -ge ${#JOBS[@]} ]; then
            echo "Error: Job number $TARGET_JOB is out of range (1-${#JOBS[@]})."
            exit 1
        fi
    else
        for i in "${!JOBS[@]}"; do
            if [ "${JOBS[$i]}" = "$TARGET_JOB" ]; then
                JOB_INDEX=$i
                break
            fi
        done
        if [ $JOB_INDEX -eq -1 ]; then
            echo "Error: Job name '$TARGET_JOB' not found. Run with --list to see available jobs."
            exit 1
        fi
    fi
    run_job_by_index $JOB_INDEX
    exit 0
fi

# 3. Run Specific Model Mode
if [ "$MODE" = "model" ]; then
    MATCH_COUNT=0
    for i in "${!JOBS[@]}"; do
        if [[ "${JOBS[$i]}" == "${TARGET_MODEL}_"* ]]; then
            run_job_by_index $i
            MATCH_COUNT=$((MATCH_COUNT + 1))
        fi
    done
    if [ $MATCH_COUNT -eq 0 ]; then
        echo "Error: No jobs matched model prefix '$TARGET_MODEL'."
        exit 1
    fi
    echo "All jobs for model '$TARGET_MODEL' completed!"
    exit 0
fi

# 4. Run Specific Task Mode
if [ "$MODE" = "task" ]; then
    MATCH_COUNT=0
    for i in "${!JOBS[@]}"; do
        if [[ "${JOBS[$i]}" == *"${TARGET_TASK}"* ]]; then
            run_job_by_index $i
            MATCH_COUNT=$((MATCH_COUNT + 1))
        fi
    done
    if [ $MATCH_COUNT -eq 0 ]; then
        echo "Error: No jobs matched task '$TARGET_TASK'. Choose 'patching' or 'ablation'."
        exit 1
    fi
    echo "All jobs for task '$TARGET_TASK' completed!"
    exit 0
fi

# 5. Run All Mode
if [ "$MODE" = "all" ]; then
    echo "========================================================================"
    echo "Starting Sequentially All ${#JOBS[@]} Phase C Jobs across 8 Models"
    echo "========================================================================"
    for i in "${!JOBS[@]}"; do
        run_job_by_index $i
    done
    echo "========================================================================"
    echo "All Phase C runs completed successfully!"
    echo "Results stored in: v1/results/derived/v1_phase_c/"
    echo "========================================================================"
    exit 0
fi
