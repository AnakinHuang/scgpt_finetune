#!/bin/bash
#SBATCH -J setup_env
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /scratch/group/p.nairr250244.000/scGPT/logs/%x-%j.out
#SBATCH -e /scratch/group/p.nairr250244.000/scGPT/logs/%x-%j.err

set -euo pipefail

WORKDIR=/scratch/group/p.nairr250244.000/scGPT
LOGDIR=$WORKDIR/logs
mkdir -p "$LOGDIR"
cd "$WORKDIR"

# Load modules
module load Anaconda3/2024.02-1 CUDA/11.7.0 cuDNN/8.5.0.96-CUDA-11.7.0

# Enable 'conda activate'
source /sw/eb/sw/Anaconda3/2024.02-1/etc/profile.d/conda.sh

# Create or update the environment
ENV_YML=docs/environment.yml
ENV_NAME=scgpt_finetune

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Updating existing env: $ENV_NAME"
  conda env update -n "$ENV_NAME" -f "$ENV_YML" --prune
else
  echo "Creating env: $ENV_NAME"
  conda env create -n "$ENV_NAME" -f "$ENV_YML"
fi

conda activate "$ENV_NAME"

# Extra requirements
pip install -r docs/requirements_2.txt

# Sanity check: print Python + Torch info to logs
python - <<'PY'
import sys
print("Python:", sys.version)
try:
    import torch
    print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("Torch not available:", e)
PY

echo "Environment setup complete."
