#!/bin/bash
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out=analysis/results/oracle_gap.jsonl
for spec in "Heart 1 1,1" "Heart 1 0.5,1" "Heart 3 0.5,1" "Credit 1 1,1,0,0" "Credit 1 1,1,1,1" "COMPAS 1 0,1,0.75,0" "COMPAS 1 1,1,1,1" "Communities 1 1,1,1,1"; do
  set -- $spec
  python analysis/oracle_gap.py $1 $2 $3 500 4235255 150 >> $out 2>/dev/null &
done
wait
