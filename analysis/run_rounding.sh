#!/bin/bash
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out=analysis/results/rounding_impact.jsonl
for TAU in 1 3; do
 for EX in 0 1; do
  for spec in "Heart 1,1" "Heart 0.5,1" "Credit 1,1,0,0" "COMPAS 0,1,0.75,0" "Communities 1,1,1,1"; do
    set -- $spec
    EXACT_RESPONSE=$EX python analysis/response_rounding_impact.py $1 $TAU $2 1000 >> $out 2>/dev/null &
  done
  wait
 done
done
