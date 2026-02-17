#!/bin/bash -eux
#SBATCH --job-name=lbe11_stability
#SBATCH --account sci-demelo-computer-vision
#SBATCH --nodelist gx01,gx03,gx04,gx05,gx06,gx25,gx28
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu-batch
#SBATCH --cpus-per-task 32
#SBATCH --mem 60000
#SBATCH --time 48:00:00
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user vincent.eichhorn@student.hpi.uni-potsdam.de
#SBATCH --output /sc/home/vincent.eichhorn/jaguar-reid/_jobs/job_lbe11_stability-%j.log

export PATH="/sc/home/vincent.eichhorn/conda3/bin:$PATH"
cd /sc/home/vincent.eichhorn/jaguar-reid
poetry install 
poetry run python -m jaguar.experiments.lbe11_stability