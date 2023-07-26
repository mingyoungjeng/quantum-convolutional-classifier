#!/bin/bash
#SBATCH --job-name=qcc                     # Job name
#SBATCH --partition=sixhour                 # Partition Name (Required)
#SBATCH --mail-type=END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mingyoungjeng@ku.edu    # Where to send mail
#SBATCH --ntasks=1                          # Run on a single CPU
#SBATCH --mem=30g                          # Job memory request
#SBATCH --time=0-06:00:00                   # Time limit days-hrs:min:sec
#SBATCH --output=%j.log                     # Standard output and error log
#SBATCH --gres=gpu                          # 1 GPU
#SBATCH --constraint=a100                   # a100, q8000, q6000, v100

pwd; hostname; date
 
module purge
module load conda/latest

# conda remove -n qcc --all -y
# conda create -n qcc python=3.10.9 -y
conda activate qcc

# conda install -c conda-forge pennylane -y
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install -e .

echo "running QCC"
cd /home/m174j393/work/qcc/test
qcc load $1

date
