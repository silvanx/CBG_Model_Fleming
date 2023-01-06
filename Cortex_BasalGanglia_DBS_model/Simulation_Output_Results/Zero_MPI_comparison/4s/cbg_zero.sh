#!/bin/bash -l
#SBATCH --job-name=18zero
# speficity number of nodes 
#SBATCH -N 1

# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node 18

# specify the walltime e.g 20 mins
#SBATCH -t 10:00:00

# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jakub.orlowski@ucd.ie

module load openmpi/3.1.4
source ~jorlowski/neuron_python/bin/activate
export PYTHONUNBUFFERED=yes
export PYNN_OUTPUT_DIRNAME=~jorlowski/zero_comparison/ZeroController-$(date +"%Y%m%d%H%M%S")_mpi${SLURM_NPROCS}

cd ~jorlowski/CBG_Model_Fleming/Cortex_BasalGanglia_DBS_model

# command to use
mpirun -n ${SLURM_NPROCS} ./run_model.py -o ${PYNN_OUTPUT_DIRNAME} ~jorlowski/zero_comparison/configs/1.yml
