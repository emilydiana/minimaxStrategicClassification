#!/bin/bash
# Runs the deployed-guarantee analysis over the paper's datasets and budget profiles.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run() { python analysis/deployed_guarantee.py "$@" 3000 $SEED >> analysis/results/deployed_$1.jsonl 2>/dev/null; }
for SEED in 4235255 4235256 4235257; do
  for TAU in 1 3 6; do
    run Heart $TAU 1,1 & run Heart $TAU 1,0 & run Heart $TAU 0.5,1 &
    run Credit $TAU 1,1,1,1 & run Credit $TAU 1,1,0,0 & run Credit $TAU 1,0.5,0,0 &
    run COMPAS $TAU 1,1,1,1 & run COMPAS $TAU 0,1,0.75,0 & run COMPAS $TAU 1,1,0,0 &
    run Communities $TAU 1,1,1,1 &
    wait
  done
done
