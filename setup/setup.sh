#!/usr/bin/env bash

# Environment
env=${1:-chaar}
requirements=${2:-requirements.txt}
packages=(${3:-haarpy})
system=${4:-cpu}
envs=${5:-${HOME}/envs}
test=${6:-test.py}

# Load modules
case ${system} in
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
virtualenv --no-download ${envs}/${env}


# Activate environment
source ${envs}/${env}/bin/activate


# Install environment
options=(--no-index $(grep -ivE "${packages[@]}" ${requirements}))
pip install ${options[@]}

options=(${packages[@]})
pip install ${options[@]}


# Test environment
if [[ -f ${test} ]]
then
	options=(-rA -W ignore::DeprecationWarning)
	pytest ${options[@]} ${test}
	rm -rf __pycache__ .pytest_cache
fi