#!/bin/bash
#SBATCH --job-name=v1_phase_c
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-16
#SBATCH --output=v1/results/logs/slurm/phase_c_%A_%a.out
#SBATCH --error=v1/results/logs/slurm/phase_c_%A_%a.err

# ==============================================================================
# Slurm Job Array Script for V1 Phase C (16 Sub-Jobs)
# Partition: h100 | CPUs: 1 | GPUs: 1 | Memory & Time: Unspecified (Cluster defaults)
# Array Range: 1-16 (Maps directly to the 16 defined Phase C tasks)
# ==============================================================================

# Resolve project root robustly under Slurm
if [ -n "$SLURM_SUBMIT_DIR" ] && [ -d "$SLURM_SUBMIT_DIR/.venv" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"

# Ensure log directory exists
mkdir -p v1/results/logs/slurm

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "Error: Virtual environment .venv not found in $PROJECT_ROOT."
    exit 1
fi

# Prevent PyTorch CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT/v1/src:$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "========================================================================"
echo "Starting Slurm Task: Array ID ${SLURM_ARRAY_TASK_ID} / Job ID ${SLURM_JOB_ID}"
echo "Running on Host: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CUDA Device')"
echo "========================================================================"

# Dispatch to run_all_phase_c.sh with the specific job ID
# Any trailing arguments passed to sbatch (e.g., --all-layers) will be forwarded
bash v1/scripts/run_all_phase_c.sh --job "$SLURM_ARRAY_TASK_ID" "$@"

echo "========================================================================"
echo "Finished Slurm Task: Array ID ${SLURM_ARRAY_TASK_ID}"
echo "========================================================================"
