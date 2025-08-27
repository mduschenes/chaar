#!/usr/bin/env bash

# Environment
env=${1:-env}
requirements=${2:-requirements.txt}
architecture=${3:-cpu}
envs=${4:-${HOME}/envs}
test=${5:-test.py}

# Load modules
case ${architecture} in
	cpu)
		modules=(python)
	;;
	gpu)
		modules=(python cuda cudnn)
	;;
	*)
	modules=(python)
	;;
esac

module purge &>/dev/null
module load ${modules[@]}


# Setup environment
dir=$(dirname ${env})
env=$(basename ${env})
if [[ ! ${dir} == . ]];then envs=${dir};fi

mkdir -p ${envs}
deactivate &>/dev/null 2>&1
rm -rf ${envs}/${env}
pip install --upgrade pip --no-index
virtualenv --no-download ${envs}/${env}

# Activate environment
source ${envs}/${env}/bin/activate

# Install environment
options=()
options+=(--no-index)
options+=(-r ${requirements})

pip install ${options[@]}

# Test environment
if [[ -f ${test} ]]
then
	options=(-rA -W ignore::DeprecationWarning)
	pytest ${options[@]} ${test}
	rm -rf __pycache__ .pytest_cache
fi