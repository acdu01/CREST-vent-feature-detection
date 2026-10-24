#!/usr/bin/env bash
#SBATCH --job-name=crest-reconstruction
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=crest-%j.log
# Add your site's account/partition/GPU directives before submitting.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate
config="${1:-configs/paper.yaml}"
python -m crest_pipeline doctor --config "$config"
python -m crest_pipeline run --config "$config"
