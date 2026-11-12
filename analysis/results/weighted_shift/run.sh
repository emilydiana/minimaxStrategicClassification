#!/bin/bash
# args: kind(main|pareto) dataset fracs taus seedidx
W=/tmp/claude-0/-home-user/8ef0713e-a8d8-5c0b-b614-2f56ec7913b9/scratchpad/wt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PLOTDIR=$W/plots_$$ SHIFT_CRITERION=weighted
cd /home/user/minimaxstrategicclassification
if [ "$1" = pareto ]; then export N_GAMMAS=10; fi
SEED_OFFSET=$5 python analysis/regen_figures.py $2 $3 $4 1 1000 $W/$1.jsonl > /dev/null 2>&1
rm -rf $W/plots_$$
