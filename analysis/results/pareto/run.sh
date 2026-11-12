#!/bin/bash
# args: version dataset fracs tau seedidx
R=/tmp/claude-0/-home-user/8ef0713e-a8d8-5c0b-b614-2f56ec7913b9/scratchpad/pareto
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 N_GAMMAS=10 PLOTDIR=$R/plots_$$
if [ "$1" = new ]; then cd /home/user/minimaxstrategicclassification; script=analysis/regen_figures.py; else cd /home/user/mssc-main; script=regen_figures.py; fi
SEED_OFFSET=$5 python $script $2 $3 $4 1 1000 $R/$1.jsonl > /dev/null 2>&1
rm -rf $R/plots_$$
