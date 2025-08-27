#!/usr/bin/env bash

#SBATCH --account=def-raymond
#SBATCH --job-name=matrix
#SBATCH --output=scratch/lanl/matrix/%x.%A.stdout
#SBATCH --error=scratch/lanl/matrix/%x.%A.stderr

##SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --mem=128G
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

path=scratch/lanl/matrix
data=
simulations=(${simulations:-${@:-"t.k.n"}})
t=(${t})
k=(${k})
n=(${n})

for simulation in ${simulations[@]}
do
	case ${simulation} in
		
		"t.2.4")
		T=(2 3 4)
		K=(1 2 3)
		N=(0 2)
		;;
		
		"t.5")
		T=(5)
		K=(1)
		N=(0 2)
		;;
		
		"t.5.k")
		T=(5)
		K=(2 3)
		N=(2)
		;;

		"t.4")
		T=(4)
		K=(1 3)
		N=(0 2)
		;;

		"t.k.n")
		T=(${t})
		K=(${k})
		N=(${n})
		;;

		*)
		T=()
		K=()
		N=()
		;;
	esac        


	for t in ${T[@]}
	do

		if [[ ! -f ${path}/data.${t}.pkl ]]
		then
			exe=./main.py
			args=(${path} ${t})
			
			echo ${exe} ${args[@]}
			${exe} ${args[@]}
		fi

		for k in ${K[@]}
		do
			for n in ${N[@]}
			do

				exe=./plot.py
				args=(${path} ${t} ${k} ${n})
			  
				echo ${exe} ${args[@]}
				${exe} ${args[@]}

				if [ ! -z ${data} ] && [ -d ${data} ]
				then
					case ${t} in
						4)
						case ${n} in
							0)
								echo cp ${path}/plot.${t}.${k}.${n}.pdf ${data}/plot.haar.coefficients.${t}.${k}.pdf
								cp ${path}/plot.${t}.${k}.${n}.pdf ${data}/plot.haar.coefficients.${t}.${k}.pdf
							;;
							2)
								echo cp ${path}/plot.${t}.${k}.${n}.pdf ${data}/plot.chaar.coefficients.${t}.${k}.pdf
								cp ${path}/plot.${t}.${k}.${n}.pdf ${data}/plot.chaar.coefficients.${t}.${k}.pdf
							;;
							*)
							;;
						esac
						;;

						*)
						;;
					esac        
				fi
			done
		done
	done

done