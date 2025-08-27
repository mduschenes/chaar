#!/usr/bin/env bash

#SBATCH --account=def-raymond
#SBATCH --job-name=basis
#SBATCH --output=scratch/lanl/basis/%x.%A.stdout
#SBATCH --error=scratch/lanl/basis/%x.%A.stderr

##SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2

#SBATCH --parsable
#SBATCH --get-user-env

#SBATCH --signal=B:SIGTERM@30


requeue()
{
	scontrol requeue ${SLURM_JOB_ID}
	exit 0
}

# trap requeue SIGTERM


# Environment
env=lanl
modules=()

module purge &>/dev/null 2>&1
module load ${modules[@]} &>/dev/null 2>&1
# conda activate ${env} &>/dev/null 2>&1
source ${HOME}/envs/${env}/bin/activate &>/dev/null 2>&1


# Program

exe=./main.py
path=scratch/lanl/basis
simulations=(${simulations:-"t.d"})
t=${t}
d=${d}

for simulation in ${simulations[@]}
do
   
	case ${simulation} in
		
		"t.5.d.5")
			T=(5)
			D=(5)
		;;
		
		"t.4.d.7")
			T=(4)
			D=(7)
		;;

		"t.d")
			T=(${t})
			D=(${d})
		;;

		*)
			T=()
			D=()
		;;
	esac    

	for t in ${T[@]}
	do
		
		for d in ${D[@]}
		do

		args=(${path} ${t} ${d})
		
		echo ${exe} ${args[@]}
		${exe} ${args[@]}

		done

	done

done