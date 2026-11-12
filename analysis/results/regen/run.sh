#!/bin/bash
# args: version(new|old) dataset fracs seedidx
S=/tmp/claude-0/-home-user/8ef0713e-a8d8-5c0b-b614-2f56ec7913b9/scratchpad/regen
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PLOTDIR=$S/plots_$1_$4_$$
if [ "$1" = new ]; then cd /home/user/minimaxstrategicclassification; script=analysis/regen_figures.py; else cd /home/user/mssc-main; script=regen_figures.py; fi
SEED_OFFSET=$4 python $script $2 $3 0,1,2,3,4,5,6 1 1000 $S/$1.jsonl > /dev/null 2>&1
rm -rf $S/plots_$1_$4_$$
