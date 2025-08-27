#!/usr/bin/env bash

t=${1}
d=${2}

job=job.slurm
options=(--export="t=${t},d=${d}")

sbatch ${options[@]} < ${job}