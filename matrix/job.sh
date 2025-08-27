#!/usr/bin/env bash

t=${1}
k=${2:-1}
n=${3:-2}

job=job.slurm
options=(--export="t=${t},k=${k},n=${n}")

sbatch ${options[@]} < ${job}