#!/bin/bash
#SBATCH --job-name=pythia_bench
#SBATCH --partition=gpu_5070ti,gpu_3080ti,gpu_3080ti_short,gpu_5080,gpu_5090,gpu_6000D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/home/zaman/Projects/multimodal-ppi/phase_c/pythia_bench_%j.log
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python /home/zaman/Projects/multimodal-ppi/phase_c/teacher_pythia_ppi.py \
  --skempi /home/zaman/Projects/ppi-data/skempi/skempi_v2.csv \
  --pdb_dir /home/zaman/Projects/ppi-data/skempi/PDBs \
  --n_complexes 40 --device cuda \
  --out /home/zaman/Projects/ppi-data/phase_c/teacher_validation_pythia_ppi.csv
